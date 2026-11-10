"""
MLCommons Keyword Spotting (Speech Commands) Benchmark

Evaluates the quantized TensorFlow Lite Keyword Spotting model using the Google Speech Commands benchmark dataset.

Modified by:
Sean Botsheane
"""

import os
import sys
import time
import struct
import serial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import get_dataset as kws_data
import kws_util

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.preprocessing import label_binarize

# Allow the benchmark to import shared helper modules from the training folder.
training_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if training_folder not in sys.path:
    sys.path.insert(0, training_folder)

from results_helper import save_benchmark_results
from esp32_energy_helper import estimate_esp32_energy_and_emissions

# Device information
device_name = "ESP32-S3"
workload_type = "Keyword Spotting"

# Number of Speech Commands samples to benchmark
NUM_SAMPLES = 1000

# Minimum measurement duration for ESP32-S3 inference timing
MIN_MEASUREMENT_TIME = 30.0

# TensorFlow Lite model
MODEL_PATH = "trained_models/kws_ref_model.tflite"

# Folder for benchmark outputs
RESULTS_FOLDER = "results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Save the benchmark output to a text file while still displaying it in the terminal.
log_path = os.path.join(RESULTS_FOLDER, "benchmark_log.txt")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for file in self.files:
            file.write(message)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()



# Serial connection used to communicate with the ESP32-S3 benchmark firmware
SERIAL_PORT = "COM8"
SERIAL_BAUD = 921600
SERIAL_TIMEOUT = 5.0

# Input tensor configuration for the quantized KWS DS-CNN model
INPUT_SHAPE = (1, 49, 10, 1)
INPUT_SIZE_BYTES = 49 * 10
INPUT_SCALE = 0.5847029089927673
INPUT_ZERO_POINT = 83

# Output tensor configuration for the 12 Speech Commands classes
OUTPUT_SIZE = 12
OUTPUT_SCALE = 0.00390625
OUTPUT_ZERO_POINT = -128

# Verify that the host model file matches the model embedded in the firmware
EXPECTED_MODEL_SIZE_BYTES = 53936


def read_exact(ser, size):
    data = bytearray()

    while len(data) < size:
        chunk = ser.read(size - len(data))

        if not chunk:
            raise TimeoutError(
                f"Timed out after receiving {len(data)} of {size} bytes"
            )

        data.extend(chunk)

    return bytes(data)


def ping_esp32(ser):
    ser.write(b"PING")
    ser.flush()

    response = read_exact(ser, 4)

    if response != b"PONG":
        raise RuntimeError(
            f"Expected PONG but received {response!r}"
        )


def run_esp32_inference(ser, input_data):
    input_data = np.ascontiguousarray(
        input_data,
        dtype=np.int8
    )

    if input_data.shape != INPUT_SHAPE:
        raise ValueError(
            f"Expected {INPUT_SHAPE}, got {input_data.shape}"
        )

    payload = input_data.tobytes()

    if len(payload) != INPUT_SIZE_BYTES:
        raise ValueError(
            f"Expected {INPUT_SIZE_BYTES} bytes, got {len(payload)}"
        )

    ser.write(b"INFR")
    ser.write(payload)
    ser.flush()

    header = read_exact(ser, 4)

    if header == b"ERR!":
        raise RuntimeError(
            "ESP32-S3 reported an inference failure"
        )

    if header != b"RSLT":
        raise RuntimeError(
            f"Expected RSLT but received {header!r}"
        )

    inference_us = struct.unpack(
        "<q",
        read_exact(ser, 8)
    )[0]

    raw_output = np.frombuffer(
        read_exact(ser, OUTPUT_SIZE),
        dtype=np.int8
    ).copy()

    return raw_output, inference_us



if __name__ == "__main__":

    # Parse the MLPerf Tiny benchmark configuration arguments.
    Flags, unparsed = kws_util.parse_command()

    # Define the 12 Speech Commands classes used by the benchmark.
    word_labels = [
        "Down", "Go", "Left", "No",
        "Off", "On", "Right", "Stop",
        "Up", "Yes", "Silence", "Unknown"
    ]

    num_labels = len(word_labels)

    # Load the Speech Commands dataset using the official MLPerf Tiny data loader.
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)

    print("Speech Commands dataset loaded successfully.\n")

    # Use the same first 1000 official MLPerf Tiny test samples as the CPU/GPU benchmarks.
    dataset = ds_test.unbatch().take(NUM_SAMPLES).batch(1)

    # Start saving everything printed from this point onwards.
    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    print("Dataset Information")
    print("---------------------")
    print("Dataset: Speech Commands v0.02")
    print(f"Classes: {word_labels}")
    print(f"Number of Classes: {num_labels}")
    print(f"Samples Evaluated: {NUM_SAMPLES}")
    print("---------------------\n")

    model_size = os.path.getsize(MODEL_PATH)

    if model_size != EXPECTED_MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"Unexpected model size: expected {EXPECTED_MODEL_SIZE_BYTES} bytes, "
            f"found {model_size} bytes"
        )

    MLPERF_TARGET = 0.90

    run_labels = [
        "Warm-up",
        "Run 1",
        "Run 2",
        "Run 3"
    ]

    # Prepare and quantize the official benchmark samples before inference timing begins.
    benchmark_inputs = []

    for features, label in dataset:

        input_data = features.numpy().astype(np.float32)

        if input_data.shape != INPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected KWS input shape: expected {INPUT_SHAPE}, "
                f"found {input_data.shape}"
            )

        # Use the same INT8 quantization as the existing Pi 5 and Android benchmarks.
        input_data = np.clip(
            np.round(
                input_data / INPUT_SCALE + INPUT_ZERO_POINT
            ),
            -128,
            127
        ).astype(np.int8)

        benchmark_inputs.append({
            "input_data": input_data,
            "label": int(label.numpy()[0])
        })

    if len(benchmark_inputs) != NUM_SAMPLES:
        raise RuntimeError(
            f"Expected {NUM_SAMPLES} benchmark samples, "
            f"found {len(benchmark_inputs)}"
        )

    benchmark_runs = []

    # Open the serial connection once and use it for all benchmark executions.
    with serial.Serial(
        SERIAL_PORT,
        SERIAL_BAUD,
        timeout=SERIAL_TIMEOUT
    ) as ser:

        print("Waiting for ESP32-S3 to boot...")
        time.sleep(3)

        ser.reset_input_buffer()
        ping_esp32(ser)

        print("ESP32-S3 serial connection established.")

        # Execute one warm-up run followed by three measured benchmark runs.
        for run_label in run_labels:

            print(f"\n================ {run_label.upper()} ================\n")

            # Record the complete process time separately for diagnostic purposes.
            # Serial transfer is included here but excluded from inference timing.
            benchmark_start_time = time.perf_counter()

            true_labels = []
            predicted_labels = []
            prediction_scores = []

            total_samples = 0
            repeat = 0
            total_device_inference_us = 0

            # Repeat the complete 1000-sample workload until at least 30 seconds
            # of ESP32-S3 inference time has been measured.
            while True:

                repeat += 1

                for benchmark_input in benchmark_inputs:

                    raw_output, inference_us = run_esp32_inference(
                        ser,
                        benchmark_input["input_data"]
                    )

                    total_device_inference_us += inference_us

                    # Dequantize the INT8 Softmax output produced by the model.
                    output_data = OUTPUT_SCALE * (
                        raw_output.astype(np.float32)
                        - OUTPUT_ZERO_POINT
                    )

                    # Quantization can cause the probabilities to sum slightly above
                    # or below 1.0, so renormalize them without applying Softmax again.
                    output_sum = np.sum(output_data)

                    if output_sum <= 0:
                        raise RuntimeError(
                            "Invalid KWS output: probability sum is not positive."
                        )

                    output_data = output_data / output_sum

                    predicted_class = np.argmax(
                        output_data
                    )

                    # Only retain predictions from the first workload repetition.
                    if repeat == 1:
                        true_labels.append(
                            benchmark_input["label"]
                        )

                        predicted_labels.append(
                            predicted_class
                        )

                        prediction_scores.append(
                            output_data
                        )

                        total_samples += 1

                device_measurement_time = (
                    total_device_inference_us / 1_000_000.0
                )

                if device_measurement_time >= MIN_MEASUREMENT_TIME:
                    break

            total_elapsed_time = (
                time.perf_counter() - benchmark_start_time
            )

            # This contains only the accumulated ESP32-S3 Invoke() time.
            total_measurement_time = (
                total_device_inference_us / 1_000_000.0
            )

            # Normalize to one complete 1000-sample workload.
            inference_time = (
                total_measurement_time / repeat
            )

            energy_kwh, emissions = estimate_esp32_energy_and_emissions(
                inference_time
            )

            accuracy = accuracy_score(
                true_labels,
                predicted_labels
            )

            precision = precision_score(
                true_labels,
                predicted_labels,
                average="weighted"
            )

            recall = recall_score(
                true_labels,
                predicted_labels,
                average="weighted"
            )

            f1 = f1_score(
                true_labels,
                predicted_labels,
                average="weighted"
            )

            roc_auc = roc_auc_score(
                true_labels,
                prediction_scores,
                multi_class="ovr"
            )

            throughput = (
                total_samples / inference_time
            )

            benchmark_status = (
                "PASSED"
                if accuracy >= MLPERF_TARGET
                else "FAILED"
            )

            benchmark_runs.append({
                "run": run_label,
                "device": device_name,
                "workload": workload_type,
                "dataset": "Speech Commands v0.02",
                "model": os.path.basename(MODEL_PATH),
                "samples": total_samples,
                "repetitions": repeat,
                "target_measurement_time": MIN_MEASUREMENT_TIME,
                "total_measurement_time": total_measurement_time,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "inference_time": inference_time,
                "throughput": throughput,
                "carbon": emissions,
                "target": MLPERF_TARGET,
                "status": benchmark_status,
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "prediction_scores": prediction_scores
            })

            print(f"Benchmark Repetitions:        {repeat}")
            print(f"Minimum Measurement Time:     {MIN_MEASUREMENT_TIME:.0f} seconds")
            print(f"Device Measurement Time:      {total_measurement_time:.3f} seconds")
            print(f"Total Benchmark Elapsed Time: {total_elapsed_time:.3f} seconds")
            print(f"Inference Time:               {inference_time:.3f} seconds")
            print(f"Throughput:                   {throughput:.2f} samples/second")
            print(f"Estimated Carbon Emissions:   {emissions:.3e} kgCO2e")

            if run_label == "Warm-up":
                print("Status:                        Excluded from Average")
            else:
                print("Status:                        Included in Average")

    final_run = benchmark_runs[-1]

    true_labels = final_run["true_labels"]
    predicted_labels = final_run["predicted_labels"]
    prediction_scores = final_run["prediction_scores"]

    accuracy = final_run["accuracy"]
    precision = final_run["precision"]
    recall = final_run["recall"]
    f1 = final_run["f1"]
    roc_auc = final_run["roc_auc"]

    total_samples = final_run["samples"]
    inference_time = final_run["inference_time"]
    throughput = final_run["throughput"]
    emissions = final_run["carbon"]

    MLPERF_TARGET = final_run["target"]

    print("\n---------------------")
    print("Performance on the Keyword Spotting Benchmark")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MLPerf Target:       {MLPERF_TARGET:.0%}")

    if accuracy >= MLPERF_TARGET:
        print("Benchmark Status:    PASSED")
    else:
        print("Benchmark Status:    FAILED")

    print("---------------------")

    print("Classification Report")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=word_labels
    ))
    print("---------------------")

    print("Emission Report")
    print(f"Estimated Carbon Emissions: {emissions:.3e} kgCO2e")
    print("---------------------")

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=word_labels
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(
        ax=ax,
        colorbar=False
    )

    plt.title("Keyword Spotting Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()

    confusion_matrix_path = os.path.join(
        RESULTS_FOLDER,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300
    )

    plt.close()

    print("Confusion matrix saved to:")
    print(confusion_matrix_path)
    print("---------------------")

    y_true_bin = label_binarize(
        true_labels,
        classes=range(num_labels)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_true_bin.ravel(),
        np.array(prediction_scores).ravel(),
        ax=ax,
        name="Micro-average ROC"
    )

    ax.set_title("Keyword Spotting ROC Curve")
    plt.tight_layout()

    roc_curve_path = os.path.join(
        RESULTS_FOLDER,
        "roc_curve.png"
    )

    plt.savefig(
        roc_curve_path,
        dpi=300
    )

    plt.close()

    print("ROC curve saved to:")
    print(roc_curve_path)
    print("---------------------")

    results_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels,
        "Prediction Correct": (
            np.array(true_labels)
            == np.array(predicted_labels)
        )
    })

    csv_path = os.path.join(
        RESULTS_FOLDER,
        "ESP32S3_keyword_spotting_predictions.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    save_benchmark_results(
        results_folder=RESULTS_FOLDER,
        results_filename="ESP32S3_keyword_spotting_benchmark_results.csv",
        benchmark_runs=benchmark_runs,
    )

    print("\nBenchmark results saved to:")
    print(os.path.join(
        RESULTS_FOLDER,
        "ESP32S3_keyword_spotting_benchmark_results.csv"
    ))
    print("---------------------")

    print("\n==========================================")
    print("Benchmark completed successfully.")
    print("==========================================")

    print(f"Device: {device_name}")
    print(f"Workload: {workload_type}")

    print()
    print("Warm-up Run: Completed")
    print("Measured Runs: 3")

    print()
    print("Raw Benchmark Data:")
    print("ESP32S3_keyword_spotting_benchmark_results.csv")

    print()
    print("Prediction Results:")
    print("ESP32S3_keyword_spotting_predictions.csv")

    print()
    print("Confusion Matrix:")
    print("confusion_matrix.png")

    print()
    print("ROC Curve:")
    print("roc_curve.png")

    print()
    print("The warm-up execution is excluded from the average.")
    print("Run compile_results.py to calculate the averaged benchmark results.")

    print()
    print("Inference timing is measured directly on the ESP32-S3.")
    print("Serial transfer time is excluded from inference time.")

    print()
    print("Carbon emissions are estimated using inference time and estimated ESP32-S3 power draw.")

    print("==========================================")

    sys.stdout = original_stdout
    log_file.close()
