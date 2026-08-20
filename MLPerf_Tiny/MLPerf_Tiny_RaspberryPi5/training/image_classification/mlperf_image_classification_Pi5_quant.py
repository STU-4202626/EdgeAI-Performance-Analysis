'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

test.py: performances on cifar10 test set
target performances: https://github.com/SiliconLabs/platform_ml_models/tree/master/eembc/CIFAR10_ResNetv1
'''
import os
import time
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import train
import eval_functions_eembc
from tflite_runtime.interpreter import Interpreter
from codecarbon import EmissionsTracker
from sklearn.metrics import (roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,)
import matplotlib
matplotlib.use("Agg")

# Allow the benchmark to import shared helper modules from the training folder.
training_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if training_folder not in sys.path:
    sys.path.insert(0, training_folder)

from results_helper import save_benchmark_results

# if True uses the official MLPerf Tiny subset of CIFAR10 for validation
# if False uses the full CIFAR10 validation set
PERF_SAMPLE = True

# Device information
device_name = "Raspberry Pi 5"
workload_type = "Image Classification"

# Minimum measurement duration for inference timing and CodeCarbon
MIN_MEASUREMENT_TIME = 30.0

# Folder for benchmark outputs
results_folder = "results_quant"

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

    cifar_10_dir = 'cifar-10-batches-py'

    # Load the CIFAR-10 dataset and corresponding labels
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)
    
    # Convert byte labels to readable strings
    label_names = [label.decode("utf-8") for label in label_names]
    
    # Use the official MLPerf Tiny performance sample subset for benchmarking instead of the full CIFAR-10 test set
    if PERF_SAMPLE:
        _idxs = np.load('perf_samples_idxs.npy')
        test_data = test_data[_idxs]
        test_labels = test_labels[_idxs]
        test_filenames = test_filenames[_idxs]

    label_classes = np.argmax(test_labels,axis=1)

    # Start saving everything printed from this point onwards.
    log_file = open(log_path, "w", encoding="utf-8")

    # Keep a reference to the original console output.
    original_stdout = sys.stdout

    sys.stdout = Tee(original_stdout, log_file)

    print("\nDataset Information")
    print("---------------------")
    print("Dataset: CIFAR-10")
    print(f"Classes: {label_names}")
    print(f"Number of Classes: {len(label_names)}")
    print(f"Images Evaluated: {len(test_data)}")
    print("---------------------\n")

    # Load the quantized MLPerf Tiny ResNet TensorFlow Lite model used for CIFAR-10 image classification
    MODEL_PATH = "trained_models/pretrainedResnet_quant.tflite"

    interpreter = Interpreter(model_path=MODEL_PATH)

    # Allocate memory for the model tensors before inference.
    interpreter.allocate_tensors()
 
    # Retrieve information about the model's input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Official MLPerf Tiny accuracy target used to determine whether the benchmark passes.
    MLPERF_TARGET = 0.85

    # Labels used to identify the warm-up execution and the benchmark runs.
    run_labels = [
        "Warm-up",
        "Run 1",
        "Run 2",
        "Run 3"
    ]

    # Store the benchmark results from each execution.
    benchmark_runs = []

    # Retrieve the quantization parameters required to convert between floating-point values and the INT8 TensorFlow Lite model.
    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    # Prepare the benchmark inputs before starting the measured inference runs.
    # This keeps input expansion and quantization outside the inference-time measurement.
    benchmark_inputs = []

    for image in test_data:

        input_data = np.expand_dims(
            image,
            axis=0
        )

        # Quantize the floating-point image if the model expects INT8 input.
        if input_details[0]["dtype"] == np.int8:

            input_data = np.clip(
                np.round(input_data / input_scale + input_zero_point),
                -128,
                127
            ).astype(np.int8)

        else:

            input_data = input_data.astype(np.float32)

        benchmark_inputs.append(input_data)

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

        # Start measuring the energy consumed during the current benchmark run.
        tracker.start()

        # Record the start time before inference begins.
        start_time = time.perf_counter()

        # Store the prediction probabilities produced for each benchmark image.
        predictions = []

        # Repeat the complete benchmark workload until the minimum measurement duration is reached.
        repeat = 0

        while True:

            repeat += 1

            # Store predictions only during the first repetition.
            # This prevents the classification results from being duplicated.

            # Perform inference on each benchmark image individually using the TensorFlow Lite interpreter.
            for input_data in benchmark_inputs:

                # Copy the image into the TensorFlow Lite input tensor.
                interpreter.set_tensor(
                    input_details[0]["index"],
                    input_data
                )

                # Execute inference using the TensorFlow Lite interpreter.
                interpreter.invoke()

                # Retrieve the model output.
                output = interpreter.get_tensor(
                    output_details[0]["index"]
                )

                # Convert the INT8 output tensor back to floating-point probabilities.
                if output_details[0]["dtype"] == np.int8:

                    output = output_scale * (
                        output.astype(np.float32)
                        - output_zero_point
                    )

                if repeat == 1:
                    predictions.append(output[0])


            # Check whether the minimum measurement duration has been reached.
            elapsed_time = time.perf_counter() - start_time

            if elapsed_time >= MIN_MEASUREMENT_TIME:
                break

        # Calculate the total measurement time for the current benchmark run.
        total_measurement_time = time.perf_counter() - start_time

        # Calculate the average time required to process the benchmark dataset once.
        inference_time = total_measurement_time / repeat

        # Stop measuring the carbon emissions for the current benchmark run.
        total_emission = tracker.stop()

        # Calculate the carbon emissions for one benchmark dataset execution.
        emission = total_emission / repeat

        # Convert the collected predictions into a NumPy array for metric calculations.
        predictions = np.array(predictions)

        # Convert probability predictions into the most likely class for each image.
        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        # Calculate classification accuracy using scikit-learn.
        accuracy = accuracy_score(
            label_classes,
            predicted_classes
        )

        # Calculate additional performance metrics used to evaluate the benchmark.
        precision = precision_score(
            label_classes,
            predicted_classes,
            average="weighted"
        )

        recall = recall_score(
            label_classes,
            predicted_classes,
            average="weighted"
        )

        f1 = f1_score(
            label_classes,
            predicted_classes,
            average="weighted"
        )

        # Calculate the number of benchmark images processed during one repetition.
        samples_processed = len(label_classes)

        # Calculate the average number of images processed per second.
        throughput = samples_processed / inference_time

        # Compute multi-class ROC-AUC using prediction probabilities.
        auc_scikit = roc_auc_score(
            test_labels,
            predictions
        )

        # Determine whether the benchmark satisfies the official MLPerf Tiny target.
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

            "dataset": "CIFAR-10",

            "model": os.path.basename(MODEL_PATH),

            "samples": samples_processed,

            "repetitions": repeat,

            "target_measurement_time": MIN_MEASUREMENT_TIME,

            "total_measurement_time": total_measurement_time,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": auc_scikit,

            "inference_time": inference_time,

            "throughput": throughput,

            "carbon": emission,

            "target": MLPERF_TARGET,

            "status": benchmark_status,

            "predictions": predictions,

            "predicted_classes": predicted_classes
        })

        # Display a summary of the current benchmark execution.
        print(f"Benchmark Repetitions:    {repeat}")
        print(f"Minimum Measurement Time: {MIN_MEASUREMENT_TIME:.0f} seconds")
        print(f"Total Measurement Time:   {total_measurement_time:.3f} seconds")
        print(f"Inference Time:           {inference_time:.3f} seconds")
        print(f"Throughput:               {throughput:.2f} images/second")
        print(f"Carbon Emissions:         {emission:.3e} kgCO2e")

        # Indicate whether the current execution will be used when calculating the final benchmark averages.
        if run_label == "Warm-up":
            print("Status:              Excluded from Average")
        else:
            print("Status:              Included in Average")

    # Use the final benchmark execution for the remaining reports and visualisations.
    final_run = benchmark_runs[-1]

    predictions = final_run["predictions"]

    predicted_classes = final_run["predicted_classes"]

    accuracy = final_run["accuracy"]

    precision = final_run["precision"]

    recall = final_run["recall"]

    f1 = final_run["f1"]

    auc_scikit = final_run["roc_auc"]

    samples_processed = final_run["samples"]

    inference_time = final_run["inference_time"]

    throughput = final_run["throughput"]

    emission = final_run["carbon"]

    MLPERF_TARGET = final_run["target"]
    benchmark_status = final_run["status"]

    print("\n---------------------")
    print("Performance on the Image Classification Benchmark")
    print("Accuracy:", accuracy)

    print(f"MLPerf Target:       {MLPERF_TARGET:.0%}")

    if accuracy >= MLPERF_TARGET:
        print("Benchmark Status:    PASSED")
    else:
        print("Benchmark Status:    FAILED")

    print("---------------------")

    # Generate a detailed classification report containing: Precision, Recall, F1-Score, and Support for each class
    print("Classification Report")
    print(classification_report(label_classes, predicted_classes))
    print("---------------------")

    # Generate and display the confusion matrix
    # Rows represent the true classes and columns represent the predicted classes.
    cm = confusion_matrix(label_classes, predicted_classes)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names
    )

    disp.plot(
        cmap="Blues",
        xticks_rotation=45
    )

    plt.title("MLPerf Tiny Image Classification - Confusion Matrix")
    plt.tight_layout()

    # Save the figure for use in the dissertation
    confusion_matrix_path = os.path.join(
        results_folder,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Confusion matrix saved to:")
    print(confusion_matrix_path)
    print("---------------------")

    plt.close()

    # Display the measured carbon emissions produced during benchmarking.
    print("Emission Report")
    print(f"Carbon Emissions:     {emission:.3e} kgCO2e")
    print("---------------------")

    # Calculate accuracy using the official EEMBC evaluation function to compare against the scikit-learn result
    print("EEMBC calculate_accuracy method")
    accuracy_eembc = eval_functions_eembc.calculate_accuracy(predictions, label_classes)
    print("---------------------")

    # Compute multi-class ROC-AUC using prediction probabilities.
    # ROC-AUC evaluates the model's ability to distinguish between different CIFAR-10 classes.
    print("sklearn.metrics.roc_auc_score method")
    print("AUC sklearn: ", auc_scikit)
    print("---------------------")

    # Compute ROC-AUC using the official EEMBC evaluation implementation
    print("EEMBC calculate_auc method")
    auc_eembc = eval_functions_eembc.calculate_auc(predictions, label_classes, label_names, os.path.basename(MODEL_PATH))
    
    # Save the prediction results for every benchmark image to a CSV file for further analysis
    results_df = pd.DataFrame({
        "True Label": label_classes,
        "Predicted Label": predicted_classes,
        "Prediction Correct": label_classes == predicted_classes
    })

    csv_path = os.path.join(
        results_folder,
        "Pi5_image_classification_predictions_quant.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    # Save the warm-up execution and benchmark runs to a CSV file.
    save_benchmark_results(
        results_folder=results_folder,
        results_filename="Pi5_image_classification_benchmark_results_quant.csv",
        benchmark_runs=benchmark_runs,
    )

    print("\nBenchmark results saved to:")
    print(os.path.join(
        results_folder,
        "Pi5_image_classification_benchmark_results_quant.csv"
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
    print("Pi5_image_classification_benchmark_results_quant.csv")

    print()
    print("Prediction Results:")
    print("Pi5_image_classification_predictions_quant.csv")

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