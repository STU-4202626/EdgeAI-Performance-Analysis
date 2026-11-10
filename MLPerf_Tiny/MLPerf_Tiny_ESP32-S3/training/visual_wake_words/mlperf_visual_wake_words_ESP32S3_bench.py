"""
MLCommons Tiny Visual Wake Words Benchmark

Performance evaluation of the quantized MLPerf Tiny Visual Wake Words
model on the ESP32-S3 using device-side TensorFlow Lite Micro inference.

Modified by:
Sean Botsheane
"""

import os
import random
import struct
import sys
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Allow the benchmark to import shared helper modules from the training folder.
training_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if training_folder not in sys.path:
    sys.path.insert(0, training_folder)

from results_helper import save_benchmark_results
from esp32_energy_helper import estimate_esp32_energy_and_emissions


# Device and workload information
device_name = "ESP32-S3"
workload_type = "Visual Wake Words"

# Dataset configuration
IMAGE_SIZE = (96, 96)
NUM_IMAGES = 1000

# Minimum duration measured directly from ESP32-S3 Invoke() calls.
MIN_MEASUREMENT_TIME = 30.0

# Visual Wake Words dataset folders.
DATASET_DIR = "vw_coco2014_96"
PERSON_DIR = os.path.join(DATASET_DIR, "person")
NON_PERSON_DIR = os.path.join(DATASET_DIR, "non_person")

# Quantized MLPerf Tiny Visual Wake Words model.
MODEL_PATH = "trained_models/vww_96_int8.tflite"

# Folder for benchmark outputs.
RESULTS_FOLDER = "results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Save terminal output to a log while still displaying it.
log_path = os.path.join(
    RESULTS_FOLDER,
    "benchmark_log.txt"
)


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


# Serial connection used to communicate with the ESP32-S3 benchmark firmware.
SERIAL_PORT = "COM8"
SERIAL_BAUD = 921600
SERIAL_TIMEOUT = 5.0

# Input tensor configuration for the quantized MobileNetV1 model.
INPUT_SHAPE = (1, 96, 96, 3)
INPUT_SIZE_BYTES = 96 * 96 * 3
INPUT_SCALE = 0.003921568859368563
INPUT_ZERO_POINT = -128

# Output tensor configuration for the two Visual Wake Words classes.
OUTPUT_SIZE = 2
OUTPUT_SCALE = 0.00390625
OUTPUT_ZERO_POINT = -128

# Verify that the host model file matches the model embedded in the firmware.
EXPECTED_MODEL_SIZE_BYTES = 333288


def read_exact(ser, size):
    data = bytearray()

    while len(data) < size:
        chunk = ser.read(
            size - len(data)
        )

        if not chunk:
            raise TimeoutError(
                f"Timed out after receiving {len(data)} of {size} bytes"
            )

        data.extend(chunk)

    return bytes(data)


def ping_esp32(ser):
    ser.write(b"PING")
    ser.flush()

    response = read_exact(
        ser,
        4
    )

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

    header = read_exact(
        ser,
        4
    )

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

    # Randomly sample the same balanced 500/500 dataset used by the existing VWW benchmarks.
    random.seed(42)

    person_images = random.sample(
        os.listdir(PERSON_DIR),
        NUM_IMAGES // 2
    )

    non_person_images = random.sample(
        os.listdir(NON_PERSON_DIR),
        NUM_IMAGES // 2
    )

    # Start saving everything printed from this point onwards.
    log_file = open(
        log_path,
        "w",
        encoding="utf-8"
    )

    original_stdout = sys.stdout
    sys.stdout = Tee(
        original_stdout,
        log_file
    )

    print("Dataset Information")
    print("---------------------")
    print("Dataset: Visual Wake Words")
    print("Classes: ['non_person', 'person']")
    print(f"Person images: {len(person_images)}")
    print(f"Non-person images: {len(non_person_images)}")
    print(f"Images Evaluated: {NUM_IMAGES}")
    print("---------------------\n")

    class_names = [
        "non_person",
        "person"
    ]

    # Build the same balanced evaluation dataset structure as the existing benchmarks.
    image_paths = []
    true_labels = []

    for image_name in person_images:
        image_paths.append(
            os.path.join(
                PERSON_DIR,
                image_name
            )
        )
        true_labels.append(1)

    for image_name in non_person_images:
        image_paths.append(
            os.path.join(
                NON_PERSON_DIR,
                image_name
            )
        )
        true_labels.append(0)

    # Shuffle image paths and labels together using the existing fixed random seed.
    combined = list(
        zip(
            image_paths,
            true_labels
        )
    )

    random.shuffle(combined)

    image_paths, true_labels = zip(
        *combined
    )

    image_paths = list(
        image_paths
    )

    true_labels = np.array(
        true_labels,
        dtype=np.int64
    )

    model_size = os.path.getsize(
        MODEL_PATH
    )

    if model_size != EXPECTED_MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"Unexpected model size: expected {EXPECTED_MODEL_SIZE_BYTES} bytes, "
            f"found {model_size} bytes"
        )

    # Official MLPerf Tiny accuracy target for Visual Wake Words.
    MLPERF_TARGET = 0.80

    run_labels = [
        "Warm-up",
        "Run 1",
        "Run 2",
        "Run 3"
    ]

    benchmark_runs = []

    # Prepare and quantize all images before benchmark timing begins.
    benchmark_inputs = []

    for image_path, label in zip(
        image_paths,
        true_labels
    ):
        img = Image.open(
            image_path
        ).convert("RGB")

        img = img.resize(
            IMAGE_SIZE
        )

        img_array = np.asarray(
            img,
            dtype=np.float32
        )

        img_array /= 255.0

        img_array = np.clip(
            np.round(
                img_array / INPUT_SCALE
                + INPUT_ZERO_POINT
            ),
            -128,
            127
        ).astype(np.int8)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        if img_array.shape != INPUT_SHAPE:
            raise RuntimeError(
                f"Unexpected VWW input shape: expected {INPUT_SHAPE}, "
                f"found {img_array.shape}"
            )

        benchmark_inputs.append({
            "input_data": img_array,
            "label": int(label)
        })

    if len(benchmark_inputs) != NUM_IMAGES:
        raise RuntimeError(
            f"Expected {NUM_IMAGES} benchmark images, "
            f"found {len(benchmark_inputs)}"
        )

    # Open the serial connection once and reuse it for all four executions.
    with serial.Serial(
        SERIAL_PORT,
        SERIAL_BAUD,
        timeout=SERIAL_TIMEOUT
    ) as ser:

        print("Waiting for ESP32-S3 to boot...")
        time.sleep(8)

        ser.reset_input_buffer()
        ping_esp32(ser)

        print("ESP32-S3 serial connection established.")

        # Execute one warm-up run followed by three measured benchmark runs.
        for run_label in run_labels:

            print(f"\n================ {run_label.upper()} ================\n")

            # Record complete host elapsed time separately from device inference time.
            benchmark_start_time = time.perf_counter()

            predicted_labels = []
            prediction_probabilities = []

            repeat = 0
            total_device_inference_us = 0

            # Repeat complete 1000-image workloads until at least 30 seconds
            # of ESP32-S3 Invoke() time has been accumulated.
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

                    # Quantization can move the two probabilities slightly away from a total of 1.0, so renormalize without applying Softmax again.
                    output_sum = np.sum(
                        output_data
                    )

                    if output_sum <= 0:
                        raise RuntimeError(
                            "Invalid VWW output: probability sum is not positive."
                        )

                    output_data = (
                        output_data / output_sum
                    )

                    predicted_class = int(
                        np.argmax(output_data)
                    )

                    # Only retain predictions from the first workload repetition.
                    if repeat == 1:
                        predicted_labels.append(
                            predicted_class
                        )

                        prediction_probabilities.append(
                            float(output_data[1])
                        )

                device_measurement_time = (
                    total_device_inference_us
                    / 1_000_000.0
                )

                if device_measurement_time >= MIN_MEASUREMENT_TIME:
                    break

            total_elapsed_time = (
                time.perf_counter()
                - benchmark_start_time
            )

            # Benchmark timing contains only accumulated ESP32-S3 Invoke() time.
            total_measurement_time = (
                total_device_inference_us
                / 1_000_000.0
            )

            # Normalize the measured device time to one complete 1000-image workload.
            inference_time = (
                total_measurement_time
                / repeat
            )

            predicted_labels = np.array(
                predicted_labels,
                dtype=np.int64
            )

            prediction_probabilities = np.array(
                prediction_probabilities,
                dtype=np.float32
            )

            if len(predicted_labels) != NUM_IMAGES:
                raise RuntimeError(
                    f"Expected {NUM_IMAGES} predictions, "
                    f"found {len(predicted_labels)}"
                )

            # Estimate ESP32-S3 carbon emissions for one complete workload.
            energy_kwh, emissions = estimate_esp32_energy_and_emissions(
                inference_time
            )

            accuracy = accuracy_score(
                true_labels,
                predicted_labels
            )

            precision = precision_score(
                true_labels,
                predicted_labels
            )

            recall = recall_score(
                true_labels,
                predicted_labels
            )

            f1 = f1_score(
                true_labels,
                predicted_labels
            )

            roc_auc = roc_auc_score(
                true_labels,
                prediction_probabilities
            )

            throughput = (
                NUM_IMAGES
                / inference_time
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
                "dataset": "Visual Wake Words",
                "model": os.path.basename(MODEL_PATH),
                "samples": NUM_IMAGES,
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
                "predicted_labels": predicted_labels,
                "prediction_probabilities": prediction_probabilities
            })

            print(f"Benchmark Repetitions:        {repeat}")
            print(f"Minimum Measurement Time:     {MIN_MEASUREMENT_TIME:.0f} seconds")
            print(f"Device Measurement Time:      {total_measurement_time:.3f} seconds")
            print(f"Total Benchmark Elapsed Time: {total_elapsed_time:.3f} seconds")
            print(f"Inference Time:               {inference_time:.3f} seconds")
            print(f"Throughput:                   {throughput:.2f} images/second")
            print(f"Estimated Carbon Emissions:   {emissions:.3e} kgCO2e")

            if run_label == "Warm-up":
                print("Status:                       Excluded from Average")
            else:
                print("Status:                       Included in Average")

    # Use the final measured execution for reports and saved prediction outputs.
    final_run = benchmark_runs[-1]

    predicted_labels = final_run[
        "predicted_labels"
    ]

    prediction_probabilities = final_run[
        "prediction_probabilities"
    ]

    accuracy = final_run["accuracy"]
    precision = final_run["precision"]
    recall = final_run["recall"]
    f1 = final_run["f1"]
    roc_auc = final_run["roc_auc"]

    inference_time = final_run[
        "inference_time"
    ]

    throughput = final_run[
        "throughput"
    ]

    emissions = final_run[
        "carbon"
    ]

    MLPERF_TARGET = final_run[
        "target"
    ]

    print("\n---------------------")
    print("Performance on the Visual Wake Words Benchmark")
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
        target_names=class_names
    ))
    print("---------------------")

    print("Emission Report")
    print(f"Estimated Carbon Emissions: {emissions:.3e} kgCO2e")
    print("---------------------")

    # Save the confusion matrix for the final benchmark execution.
    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        cmap="Blues"
    )

    plt.title(
        "MLPerf Tiny Visual Wake Words - Confusion Matrix"
    )

    confusion_matrix_path = os.path.join(
        RESULTS_FOLDER,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved to:")
    print(confusion_matrix_path)
    print("---------------------")

    # Save the ROC curve using the person-class probability.
    fpr, tpr, thresholds = roc_curve(
        true_labels,
        prediction_probabilities
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "MLPerf Tiny Visual Wake Words - ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    roc_curve_path = os.path.join(
        RESULTS_FOLDER,
        "roc_curve.png"
    )

    plt.savefig(
        roc_curve_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("ROC curve saved to:")
    print(roc_curve_path)
    print("---------------------")

    # Save image-level prediction results for the final benchmark execution.
    results_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels,
        "Prediction Correct": (
            true_labels
            == predicted_labels
        )
    })

    csv_path = os.path.join(
        RESULTS_FOLDER,
        "ESP32S3_visual_wake_words_predictions.csv"
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
        results_filename="ESP32S3_visual_wake_words_benchmark_results.csv",
        benchmark_runs=benchmark_runs,
    )

    print("\nBenchmark results saved to:")
    print(os.path.join(
        RESULTS_FOLDER,
        "ESP32S3_visual_wake_words_benchmark_results.csv"
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
    print("ESP32S3_visual_wake_words_benchmark_results.csv")

    print()
    print("Prediction Results:")
    print("ESP32S3_visual_wake_words_predictions.csv")

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
