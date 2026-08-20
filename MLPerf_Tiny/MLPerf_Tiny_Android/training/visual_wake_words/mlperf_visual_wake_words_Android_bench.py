'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

Visual Wake Words Benchmark

mlperf_visual_wake_words_PC_bench.py:
Performance evaluation of the MLPerf Tiny Visual Wake Words
benchmark using the official MobileNet V1 model.

Modified by:
Sean Botsheane
'''

import os
import random
import time
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_score, recall_score, f1_score)
from codecarbon import EmissionsTracker
import matplotlib
matplotlib.use("Agg")

# Allow the benchmark to import shared helper modules from the training folder.
training_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if training_folder not in sys.path:
    sys.path.insert(0, training_folder)

from results_helper import save_benchmark_results

# Device and workload information
device_name = "Samsung Galaxy S26 Ultra"
workload_type = "Visual Wake Words"

# Dataset configuration
IMAGE_SIZE = (96, 96)
NUM_IMAGES = 1000

# Minimum measurement duration for CodeCarbon and inference timing
MIN_MEASUREMENT_TIME = 30.0

# Root directory containing the Visual Wake Words dataset
DATASET_DIR = "vw_coco2014_96"

# Individual class folders
PERSON_DIR = os.path.join(DATASET_DIR, "person")
NON_PERSON_DIR = os.path.join(DATASET_DIR, "non_person")

# Load the pre-trained MLPerf Tiny model
MODEL_PATH = "trained_models/vww_96_int8.tflite"

# Visual Wake Words dataset location
dataset_path = "vw_coco2014_96"

# Folder for saving benchmark results
results_folder = "results"

os.makedirs(results_folder, exist_ok=True)

# Save the benchmark output to a text file while still displaying it in the terminal.
log_path = os.path.join(results_folder, "benchmark_log.txt")

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

if __name__ == "__main__":

    # Load the quantized TensorFlow Lite model.
    interpreter = Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    print("Input Quantization:", input_details[0]["quantization"])
    print("Output Quantization:", output_details[0]["quantization"])
    print()

    print("Input Tensor:", input_details[0]["shape"], input_details[0]["dtype"])
    print("Output Tensor:", output_details[0]["shape"], output_details[0]["dtype"])

    print("TensorFlow Lite model loaded successfully.\n")

    # Randomly sample an equal number of images from each class
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
    log_file = open(log_path, "w", encoding="utf-8")

    # Keep a reference to the original console output.
    original_stdout = sys.stdout

    sys.stdout = Tee(original_stdout, log_file)

    print("Dataset Information")
    print("---------------------")
    print("Dataset: Visual Wake Words")
    print("Classes: ['non_person', 'person']")
    print(f"Person images: {len(person_images)}")
    print(f"Non-person images: {len(non_person_images)}")
    print(f"Images Evaluated: {NUM_IMAGES}")
    print("---------------------\n")

    class_names = ["non_person", "person"]

    # Build a balanced evaluation dataset
    image_paths = []
    true_labels = []

    # Person = class 1
    for img in person_images:
        image_paths.append(os.path.join(PERSON_DIR, img))
        true_labels.append(1)

    # Non-person = class 0
    for img in non_person_images:
        image_paths.append(os.path.join(NON_PERSON_DIR, img))
        true_labels.append(0)

    # Shuffle images and labels together
    combined = list(zip(image_paths, true_labels))
    random.shuffle(combined)

    image_paths, true_labels = zip(*combined)

    image_paths = list(image_paths)
    true_labels = np.array(true_labels)

    # Official MLPerf Tiny accuracy target used to determine whether the benchmark passes.
    MLPERF_TARGET = 0.80

    # Labels used to identify the warm-up execution and the benchmark runs.
    run_labels = [
        "Warm-up",
        "Run 1",
        "Run 2",
        "Run 3"
    ]

    # Store the benchmark results from each execution.
    benchmark_runs = []

    # Prepare the benchmark inputs before starting the measured inference runs.
    # This keeps image loading, resizing, normalisation and quantisation outside the inference-time measurement.
    benchmark_inputs = []

    for image_path, label in zip(image_paths, true_labels):

        # Load and resize the image.
        img = Image.open(image_path).convert("RGB")

        img = img.resize(IMAGE_SIZE)

        # Convert the image into a NumPy array.
        img_array = np.asarray(
            img,
            dtype=np.float32
        )

        # Normalise the pixel values.
        img_array /= 255.0

        # Quantise the input for the TensorFlow Lite model.
        img_array = np.clip(
            np.round(
                img_array / input_scale + input_zero_point
            ),
            -128,
            127
        ).astype(np.int8)

        # Add the batch dimension expected by the model.
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        benchmark_inputs.append({
            "input_data": img_array,
            "label": int(label)
        })

    # Execute one warm-up run followed by three benchmark runs used to calculate the final average.
    for run_label in run_labels:

        # Create a fresh CodeCarbon tracker for this benchmark run.
        tracker = EmissionsTracker(
            project_name=f"{device_name}_{workload_type}",
            measure_power_secs=1,
            log_level="error",
            gpu_ids=[],
            output_dir=results_folder,
            output_file="emissions.csv"
        )

        print(f"\n================ {run_label.upper()} ================\n")

        # Create new containers for the current benchmark run.
        predicted_labels = []
        prediction_probabilities = []

        # Start measuring the carbon emissions produced during the current benchmark run.
        tracker.start()

        # Record the start time before inference begins in order to calculate the total inference time.
        start_time = time.perf_counter()

        # Repeat the complete benchmark workload until the minimum measurement duration has been reached.
        repeat = 0

        while True:

            repeat += 1

            # Run inference on each image in the balanced evaluation dataset
            for benchmark_input in benchmark_inputs:

                img_array = benchmark_input["input_data"]

                # Run inference
                interpreter.set_tensor(
                    input_details[0]["index"],
                    img_array
                )

                interpreter.invoke()

                prediction = interpreter.get_tensor(
                    output_details[0]["index"]
                )

                # Convert the quantized output back into floating-point values.
                prediction = (
                    prediction.astype(np.float32) - output_zero_point
                ) * output_scale

                # Convert the logits into probabilities.
                prediction = np.exp(prediction)

                prediction /= np.sum(
                    prediction,
                    axis=1,
                    keepdims=True
                )

                # Only collect predictions from the first repetition.
                # This prevents the classification results from being duplicated.
                if repeat == 1:

                    predicted_labels.append(
                        np.argmax(prediction)
                    )

                    prediction_probabilities.append(
                        prediction[0][1]
                    )

            # Check whether the minimum measurement duration has been reached.
            elapsed_time = time.perf_counter() - start_time

            if elapsed_time >= MIN_MEASUREMENT_TIME:
                break

        # Stop the benchmark timer
        total_measurement_time = time.perf_counter() - start_time

        # Calculate the average time required for one benchmark workload.
        inference_time = total_measurement_time / repeat

        # Stop CodeCarbon and calculate the carbon emissions
        total_emissions = tracker.stop()

        # Calculate carbon emissions for one benchmark workload.
        emissions = total_emissions / repeat

        # Convert the collected benchmark results into NumPy arrays
        true_labels = np.array(true_labels)

        predicted_labels = np.array(predicted_labels)

        prediction_probabilities = np.array(prediction_probabilities)

        # Calculate classification accuracy using scikit-learn
        # Accuracy = Correct Predictions / Total Predictions
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Calculate Precision
        # Precision measures how many predicted "Person" images were actually people
        precision = precision_score(true_labels, predicted_labels)

        # Calculate Recall
        # Recall measures how many actual "Person" images were correctly identified
        recall = recall_score(true_labels, predicted_labels)

        # Calculate the F1-score
        # F1-score provides a balanced measure of Precision and Recall
        f1 = f1_score(true_labels, predicted_labels)

        # Compute the Receiver Operating Characteristic Area Under the Curve (ROC-AUC)
        # ROC-AUC evaluates how well the model distinguishes between Person and Non-Person images
        roc_auc = roc_auc_score(true_labels, prediction_probabilities)

        # Calculate the benchmark throughput (images processed per second)
        throughput = len(true_labels) / inference_time

        benchmark_status = (
            "PASSED"
            if accuracy >= MLPERF_TARGET
            else "FAILED"
        )

        # Store the benchmark results from the current execution.
        benchmark_runs.append({

            "run": run_label,

            "device": device_name,

            "workload": workload_type,

            "dataset": "Visual Wake Words",

            "model": os.path.basename(MODEL_PATH),

            "samples": len(true_labels),

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

        # Display a summary of the current benchmark execution.
        print(f"Benchmark Repetitions:    {repeat}")
        print(f"Minimum Measurement Time: {MIN_MEASUREMENT_TIME:.0f} seconds")
        print(f"Total Measurement Time:   {total_measurement_time:.3f} seconds")
        print(f"Inference Time:           {inference_time:.3f} seconds")
        print(f"Throughput:               {throughput:.2f} images/second")
        print(f"Carbon Emissions:         {emissions:.3e} kgCO2e")

        # Indicate whether the current execution will be used when calculating the final benchmark averages.
        if run_label == "Warm-up":
            print("Status:              Excluded from Average")
        else:
            print("Status:              Included in Average")

    # Use the final benchmark execution for the remaining reports and visualisations.
    final_run = benchmark_runs[-1]

    predicted_labels = final_run["predicted_labels"]
    prediction_probabilities = final_run["prediction_probabilities"]

    accuracy = final_run["accuracy"]
    precision = final_run["precision"]
    recall = final_run["recall"]
    f1 = final_run["f1"]

    roc_auc = final_run["roc_auc"]

    inference_time = final_run["inference_time"]
    throughput = final_run["throughput"]

    emissions = final_run["carbon"]

    MLPERF_TARGET = final_run["target"]
    benchmark_status = final_run["status"]

    print("\n---------------------")
    print("Performance on the Visual Wake Words Benchmark")
    print(f"Accuracy: {accuracy:.4f}")

    print(f"MLPerf Target:       {MLPERF_TARGET:.0%}")

    if accuracy >= MLPERF_TARGET:
        print("Benchmark Status:    PASSED")
    else:
        print("Benchmark Status:    FAILED")

    print("---------------------")

    # Compute the confusion matrix to visualise correct and incorrect classifications
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues")

    plt.title("MLPerf Tiny Visual Wake Words - Confusion Matrix")

    # Save the confusion matrix for inclusion in the dissertation
    plt.savefig(
        os.path.join(results_folder, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved to:")
    print(os.path.join(results_folder, "confusion_matrix.png"))
    print("---------------------")

    # Compute the Receiver Operating Characteristic (ROC) curve
    fpr, tpr, thresholds = roc_curve(
        true_labels,
        prediction_probabilities
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("MLPerf Tiny Visual Wake Words - ROC Curve")

    plt.legend(loc="lower right")

    # Save the ROC curve for inclusion in the dissertation
    plt.savefig(
        os.path.join(results_folder, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("ROC curve saved to:")
    print(os.path.join(results_folder, "roc_curve.png"))
    print("---------------------")

    # Save the benchmark predictions to a CSV file for further analysis
    results_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels,
        "Prediction Correct": true_labels == predicted_labels
    })

    csv_path = os.path.join(
        results_folder,
        "Android_visual_wake_words_predictions.csv"
    )

    results_df.to_csv(csv_path, index=False)

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    # Save the warm-up execution and benchmark runs to a CSV file.
    save_benchmark_results(
        results_folder=results_folder,
        results_filename="Android_visual_wake_words_benchmark_results.csv",
        benchmark_runs=benchmark_runs,
    )

    print("\nBenchmark results saved to:")
    print(os.path.join(
        results_folder,
        "Android_visual_wake_words_benchmark_results.csv"
    ))
    print("---------------------")

    # Display a summary of the completed benchmark execution.
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
    print("Android_visual_wake_words_benchmark_results.csv")

    print()
    print("Prediction Results:")
    print("Android_visual_wake_words_predictions.csv")

    print()
    print("Confusion Matrix:")
    print("confusion_matrix.png")

    print()
    print("ROC Curve:")
    print("roc_curve.png")

    print()
    print("The warm-up execution is excluded from the average.")
    print("Run compile_results.py to calculate the averaged benchmark results.")

    print("==========================================")

    # Restore the normal console output before closing the log.
    sys.stdout = original_stdout

    # Close the log file once the benchmark has finished.
    log_file.close()