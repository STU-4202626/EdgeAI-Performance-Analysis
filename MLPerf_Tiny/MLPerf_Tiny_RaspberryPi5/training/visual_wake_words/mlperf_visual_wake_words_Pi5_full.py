"""
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

Visual Wake Words Benchmark

Performance evaluation of the MLPerf Tiny Visual Wake Words
benchmark using the official MobileNet V1 model.

Modified by:
Sean Botsheane
"""

import os
import random
import time
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import vww_model_updated
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
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
device_name = "Raspberry Pi 5"
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
MODEL_PATH = "trained_models/vww_96.h5"

# Visual Wake Words dataset location
dataset_path = "vw_coco2014_96"

# Folder for saving benchmark results
results_folder = "results_full"

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

    # Build the MobileNet V1 architecture.
    model = vww_model_updated.mobilenet_v1()

    # Load the pretrained weights.
    model.load_weights(MODEL_PATH)

    print("Pretrained weights loaded successfully.\n")

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
    # This keeps image loading and resizing outside the inference-time measurement.
    benchmark_inputs = []

    for image_path in image_paths:

        # Load and resize the image to 96x96 pixels
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)

        # Convert the image into a NumPy array
        img_array = np.asarray(
            img,
            dtype=np.float32
        )

        # Normalise pixel values to the range [0,1]
        img_array = img_array / 255.0

        # Add the batch dimension expected by the model.
        img_array = np.expand_dims(img_array, axis=0)

        benchmark_inputs.append(img_array)

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

        # Record the start time before inference begins.
        start_time = time.perf_counter()

        # Repeat the complete benchmark workload until the minimum measurement duration is reached.
        repeat = 0

        while True:

            repeat += 1

            # Run inference on each prepared image.
            for img_array in benchmark_inputs:

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                # Only collect predictions from the first repetition.
                if repeat == 1:
                    predicted_labels.append(np.argmax(prediction))
                    prediction_probabilities.append(prediction[0][1])

            # Check whether the minimum measurement duration has been reached.
            elapsed_time = time.perf_counter() - start_time

            if elapsed_time >= MIN_MEASUREMENT_TIME:
                break

        # Stop the benchmark timer.
        total_measurement_time = time.perf_counter() - start_time

        # Stop CodeCarbon.
        total_emissions = tracker.stop()

        # Convert the measured execution into one normalised benchmark workload.
        inference_time = total_measurement_time / repeat
        emissions = total_emissions / repeat

        predicted_labels = np.array(predicted_labels)
        prediction_probabilities = np.array(prediction_probabilities)

        # Calculate classification metrics from the first repetition only.
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, prediction_probabilities)

        # Calculate benchmark throughput across all measured repetitions.
        throughput = (NUM_IMAGES * repeat) / total_measurement_time

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

        # Display a summary of the current benchmark execution.
        print(f"Benchmark Repetitions:    {repeat}")
        print(f"Minimum Measurement Time: {MIN_MEASUREMENT_TIME:.0f} seconds")
        print(f"Total Measurement Time:   {total_measurement_time:.3f} seconds")
        print(f"Inference Time:           {inference_time:.3f} seconds")
        print(f"Throughput:               {throughput:.2f} images/second")
        print(f"Carbon Emissions:         {emissions:.3e} kgCO2e")

        if run_label == "Warm-up":
            print("Status:              Excluded from Average")
        else:
            print("Status:              Included in Average")

    # Use the final benchmark execution for the remaining reports and visualisations.
    final_run = benchmark_runs[-1]

    predicted_labels = final_run["predicted_labels"]
    prediction_probabilities = final_run["prediction_probabilities"]

    accuracy = final_run["accuracy"]
    roc_auc = final_run["roc_auc"]
    emissions = final_run["carbon"]
    MLPERF_TARGET = final_run["target"]

    print("\n---------------------")
    print("Performance on the Visual Wake Words Benchmark")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MLPerf Target:       {MLPERF_TARGET:.0%}")

    if accuracy >= MLPERF_TARGET:
        print("Benchmark Status:    PASSED")
    else:
        print("Benchmark Status:    FAILED")

    print("---------------------")

    # Generate a detailed classification report.
    print("Classification Report")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    ))
    print("---------------------")

    # Print out power emissions.
    print("Emission Report")
    print(f"Carbon Emissions:     {emissions:.3e} kgCO2e")
    print("---------------------")

    # Compute the confusion matrix.
    cm = confusion_matrix(true_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues")

    plt.title("MLPerf Tiny Visual Wake Words - Confusion Matrix")

    plt.savefig(
        os.path.join(results_folder, "confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved to:")
    print(os.path.join(results_folder, "confusion_matrix.png"))
    print("---------------------")

    # Compute the Receiver Operating Characteristic (ROC) curve.
    fpr, tpr, thresholds = roc_curve(
        true_labels,
        prediction_probabilities
    )

    plt.figure(figsize=(6, 6))

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

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MLPerf Tiny Visual Wake Words - ROC Curve")
    plt.legend(loc="lower right")

    plt.savefig(
        os.path.join(results_folder, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("ROC curve saved to:")
    print(os.path.join(results_folder, "roc_curve.png"))
    print("---------------------")

    # Save the benchmark predictions to a CSV file for further analysis.
    results_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels,
        "Prediction Correct": true_labels == predicted_labels
    })

    csv_path = os.path.join(
        results_folder,
        "Pi5_visual_wake_words_predictions_full.csv"
    )

    results_df.to_csv(csv_path, index=False)

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    # Save the warm-up execution and benchmark runs to a CSV file.
    save_benchmark_results(
        results_folder=results_folder,
        results_filename="Pi5_visual_wake_words_benchmark_results_full.csv",
        benchmark_runs=benchmark_runs,
    )

    print("\nBenchmark results saved to:")
    print(os.path.join(
        results_folder,
        "Pi5_visual_wake_words_benchmark_results_full.csv"
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
    print("Pi5_visual_wake_words_benchmark_results_full.csv")

    print()
    print("Prediction Results:")
    print("Pi5_visual_wake_words_predictions_full.csv")

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
    log_file.close()
