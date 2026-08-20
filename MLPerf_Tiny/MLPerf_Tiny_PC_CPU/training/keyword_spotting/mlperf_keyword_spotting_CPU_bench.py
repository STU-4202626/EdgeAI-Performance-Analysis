'''
MLCommons Tiny Benchmark
Keyword Spotting (Speech Commands)

Benchmark Script
----------------
Evaluates the quantized TensorFlow Lite Keyword Spotting model
using the Google Speech Commands dataset.

Additional metrics added:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Inference Time
- Throughput
- Carbon Emissions (CodeCarbon)
'''

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import get_dataset as kws_data
import kws_util
import sys
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)
from codecarbon import EmissionsTracker
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")

# Allow the benchmark to import shared helper modules from the training folder.
training_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if training_folder not in sys.path:
    sys.path.insert(0, training_folder)

from results_helper import save_benchmark_results

# Device information
device_name = "AMD Ryzen 7 6800H"
workload_type = "Keyword Spotting"

# Number of Speech Commands samples to benchmark
NUM_SAMPLES = 1000

# Minimum measurement duration for CodeCarbon and inference timing
MIN_MEASUREMENT_TIME = 30.0

# TensorFlow Lite model
MODEL_PATH = "trained_models/kws_ref_model"

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

if __name__ == '__main__':
    # Parse the MLPerf Tiny benchmark configuration arguments
    Flags, unparsed = kws_util.parse_command()

    # Define the 12 Speech Commands classes used by the benchmark
    word_labels = [
        "Down", "Go", "Left", "No",
        "Off", "On", "Right", "Stop",
        "Up", "Yes", "Silence", "Unknown"
    ]

    num_labels = len(word_labels)

    # Load the Speech Commands dataset using the official MLPerf Tiny data loader
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)

    print("Speech Commands dataset loaded successfully.\n")

    # Use the official MLPerf Tiny test dataset for benchmarking
    dataset = ds_test.unbatch().take(NUM_SAMPLES).batch(1)

    # Start saving everything printed from this point onwards.
    log_file = open(log_path, "w", encoding="utf-8")

    # Keep a reference to the original console output.
    original_stdout = sys.stdout

    sys.stdout = Tee(original_stdout, log_file)

    print("Dataset Information")
    print("---------------------")
    print("Dataset: Speech Commands v0.02")
    print(f"Classes: {word_labels}")
    print(f"Number of Classes: {num_labels}")
    print(f"Samples Evaluated: {NUM_SAMPLES}")
    print("---------------------\n")

    # Load the floating-point SavedModel used for the MLPerf Tiny Keyword Spotting benchmark.
    class LegacySavedModelInference:

        def __init__(self, model_dir, device="/CPU:0"):
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"SavedModel directory was not found: {model_dir}"
                )

            self.graph = tf.Graph()

            config = tf.compat.v1.ConfigProto(
                allow_soft_placement=True
            )

            self.session = tf.compat.v1.Session(
                graph=self.graph,
                config=config,
            )

            with self.graph.as_default():
                with tf.device(device):
                    tf.compat.v1.saved_model.loader.load(
                        self.session,
                        [tf.saved_model.SERVING],
                        model_dir,
                    )

                self.input_tensor = self.graph.get_tensor_by_name(
                    "serving_default_input_1:0"
                )

                self.output_tensor = self.graph.get_tensor_by_name(
                    "StatefulPartitionedCall:0"
                )

        def __call__(self, input_tensor):
            return self.session.run(
                self.output_tensor,
                feed_dict={
                    self.input_tensor: input_tensor,
                },
            )

        def close(self):
            self.session.close()

    infer = LegacySavedModelInference(
        MODEL_PATH,
        device="/CPU:0",
    )

    print("Input Tensor:", infer.input_tensor.shape, infer.input_tensor.dtype)
    print("Output Tensor:", infer.output_tensor.shape, infer.output_tensor.dtype)
    print("SavedModel loaded successfully.\n")

    # Official MLPerf Tiny accuracy target used to determine whether the benchmark passes.
    MLPERF_TARGET = 0.90

    # Labels used to identify the warm-up execution and the benchmark runs.
    run_labels = [
        "Warm-up",
        "Run 1",
        "Run 2",
        "Run 3"
    ]

    # Store the benchmark results from each execution.
    benchmark_runs = []

    # Execute one warm-up run followed by three benchmark runs used to calculate the final average.
    for run_label in run_labels:

        # Create a fresh CodeCarbon tracker for this benchmark run.
        tracker = EmissionsTracker(
            project_name=f"{device_name}_{workload_type}",
            measure_power_secs=1,
            log_level="error",
            gpu_ids=[],
            output_dir=RESULTS_FOLDER,
            output_file="emissions.csv"
        )

        print(f"\n================ {run_label.upper()} ================\n")

        # Create new containers for the current benchmark run.
        true_labels = []
        predicted_labels = []
        prediction_scores = []

        total_samples = 0

        # Start measuring the carbon emissions produced during the current benchmark run.
        tracker.start()

        # Record the start time before inference begins in order to calculate the total inference time.
        start_time = time.perf_counter()

        # Repeat the complete benchmark workload until the minimum measurement duration has been reached.
        repeat = 0

        while True:

            repeat += 1

            # Run inference on each Speech Commands test sample
            for features, label in dataset:

                # Convert the TensorFlow tensor into a NumPy array
                input_data = features.numpy()

                # Convert the input to float32 for the SavedModel
                input_data = input_data.astype(np.float32)

                # Run inference
                output_data = infer(input_data)

                # Convert the SavedModel output logits into probabilities using Softmax
                output_data = tf.nn.softmax(
                    np.asarray(output_data),
                    axis=1
                ).numpy()

                # Determine the predicted speech command
                predicted_class = np.argmax(output_data[0])

                # Only collect predictions from the first repetition.
                # This prevents the classification results from being duplicated.
                if repeat == 1:

                    # Store the true label
                    true_labels.append(label.numpy()[0])

                    # Store the predicted label
                    predicted_labels.append(predicted_class)

                    # Store the prediction probabilities for ROC-AUC calculation
                    prediction_scores.append(output_data[0])

                    # Count the samples in one benchmark workload.
                    total_samples += 1

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

        # Calculate classification accuracy using scikit-learn
        # Accuracy = Correct Predictions / Total Predictions
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Calculate Precision, Recall and F1-Score using weighted averaging
        precision = precision_score(true_labels, predicted_labels, average="weighted")
        recall = recall_score(true_labels, predicted_labels, average="weighted")
        f1 = f1_score(true_labels, predicted_labels, average="weighted")

        # Compute the multi-class ROC-AUC score using the prediction probabilities
        roc_auc = roc_auc_score(
            true_labels,
            prediction_scores,
            multi_class="ovr"
        )

        # Calculate the benchmark throughput (samples processed per second)
        throughput = total_samples / inference_time

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

        # Display a summary of the current benchmark execution.
        print(f"Benchmark Repetitions:    {repeat}")
        print(f"Minimum Measurement Time: {MIN_MEASUREMENT_TIME:.0f} seconds")
        print(f"Total Measurement Time:   {total_measurement_time:.3f} seconds")
        print(f"Inference Time:           {inference_time:.3f} seconds")
        print(f"Throughput:               {throughput:.2f} samples/second")
        print(f"Carbon Emissions:         {emissions:.3e} kgCO2e")

        # Indicate whether the current execution will be used when calculating the final benchmark averages.
        if run_label == "Warm-up":
            print("Status:              Excluded from Average")
        else:
            print("Status:              Included in Average")

    # Use the final benchmark execution for the remaining reports and visualisations.
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
    benchmark_status = final_run["status"]

    print("\n---------------------")
    print("Performance on the Keyword Spotting Benchmark")
    print(f"Accuracy: {accuracy:.4f}")

    print(f"MLPerf Target:       {MLPERF_TARGET:.0%}")

    if accuracy >= MLPERF_TARGET:
        print("Benchmark Status:    PASSED")
    else:
        print("Benchmark Status:    FAILED")

    print("---------------------")

    # Generate a detailed classification report containing Precision, Recall,
    # F1-Score and Support for each speech command
    print("Classification Report")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=word_labels
    ))
    print("---------------------")

    # Print the total carbon emissions measured during inference
    print("Emission Report")
    print(f"Carbon Emissions:    {emissions:.3e} kgCO2e")
    print("---------------------")

    # Generate the confusion matrix for the Keyword Spotting predictions
    cm = confusion_matrix(true_labels, predicted_labels)

    # Display and save the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=word_labels
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.title("Keyword Spotting Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()

    confusion_matrix_path = os.path.join(
        RESULTS_FOLDER,
        "confusion_matrix.png"
    )

    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    print("Confusion matrix saved to:")
    print(confusion_matrix_path)
    print("---------------------")

    # Convert the true labels into a one-hot representation for multi-class ROC calculation
    y_true_bin = label_binarize(
        true_labels,
        classes=range(num_labels)
    )

    # Plot the One-vs-Rest ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_true_bin.ravel(),
        np.array(prediction_scores).ravel(),
        ax=ax,
        name="Micro-average ROC",
        color="darkorange"
    )

    ax.set_title("Keyword Spotting ROC Curve")

    plt.tight_layout()

    roc_curve_path = os.path.join(
        RESULTS_FOLDER,
        "roc_curve.png"
    )

    plt.savefig(roc_curve_path, dpi=300)
    plt.close()

    print("ROC curve saved to:")
    print(roc_curve_path)
    print("---------------------")

    # Save the benchmark predictions to a CSV file for further analysis
    results_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels,
        "Prediction Correct": np.array(true_labels) == np.array(predicted_labels)
    })

    csv_path = os.path.join(
        RESULTS_FOLDER,
        "CPU_keyword_spotting_predictions.csv"
    )

    results_df.to_csv(csv_path, index=False)

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    # Save the warm-up execution and benchmark runs to a CSV file.
    save_benchmark_results(
        results_folder=RESULTS_FOLDER,
        results_filename="CPU_keyword_spotting_benchmark_results.csv",
        benchmark_runs=benchmark_runs,
    )

    print("\nBenchmark results saved to:")
    print(os.path.join(
        RESULTS_FOLDER,
        "CPU_keyword_spotting_benchmark_results.csv"
    ))
    print("---------------------")

    # Display the final benchmark results for the evaluated device
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
    print("CPU_keyword_spotting_benchmark_results.csv")

    print()
    print("Prediction Results:")
    print("CPU_keyword_spotting_predictions.csv")

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

    infer.close()