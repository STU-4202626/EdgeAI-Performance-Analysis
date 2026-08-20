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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)
from codecarbon import EmissionsTracker
from sklearn.preprocessing import label_binarize

# Device information
device_name = "Samsung_Galaxy_S26_Ultra"
workload_type = "Keyword Spotting"

# Number of Speech Commands samples to benchmark
NUM_SAMPLES = 1000

# TensorFlow Lite model
MODEL_PATH = "trained_models/kws_ref_model.tflite"

# Folder for benchmark outputs
RESULTS_FOLDER = "results"

os.makedirs(RESULTS_FOLDER, exist_ok=True)


num_classes = 12 # should probably draw this directly from the dataset.
# FLAGS = None

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

    print("Dataset Information")
    print("---------------------")
    print(f"Classes: {word_labels}")
    print(f"Number of Classes: {num_labels}")
    print(f"Samples Evaluated: {NUM_SAMPLES}")
    print("---------------------\n")

    # Load the quantized TensorFlow Lite model used for the MLPerf Tiny Keyword Spotting benchmark
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

    # Allocate memory for all model tensors before inference
    interpreter.allocate_tensors()
    
    # Retrieve the model input and output tensor information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nInput Quantization:", input_details[0]["quantization"])
    print("Output Quantization:", output_details[0]["quantization"])
    print()

    print("Input Tensor:", input_details[0]["shape"], input_details[0]["dtype"])
    print("Output Tensor:", output_details[0]["shape"], output_details[0]["dtype"])


    print("TensorFlow Lite model loaded successfully.\n")

    # Create lists to store the benchmark results
    true_labels = []
    predicted_labels = []
    prediction_scores = []

    total_samples = 0

    # Initialise CodeCarbon to measure the carbon emissions produced during inference
    tracker = EmissionsTracker(
        project_name=f"{device_name}_{workload_type}",
        measure_power_secs=1,
        log_level="error",
        gpu_ids=[],
        output_dir=RESULTS_FOLDER,
        output_file="emissions.csv"
    )

    tracker.start()

    # Start timing the benchmark inference
    start_time = time.perf_counter()

    # Run inference on each Speech Commands test sample
    for features, label in dataset:

        # Convert the TensorFlow tensor into a NumPy array
        input_data = features.numpy()

        # Quantize the floating-point input using the model's quantization parameters
        input_scale, input_zero_point = input_details[0]["quantization"]

        input_data = np.round(
            input_data / input_scale + input_zero_point
        ).astype(np.int8)

        # Set the input tensor for the TensorFlow Lite interpreter
        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        # Run inference
        interpreter.invoke()

        # Retrieve the prediction probabilities from the output tensor
        output_data = interpreter.get_tensor(
            output_details[0]["index"]
        )

        # Dequantize the model output back to floating-point logits
        output_scale, output_zero_point = output_details[0]["quantization"]

        output_data = output_scale * (
            output_data.astype(np.float32) - output_zero_point
        )

        # Convert the logits into probabilities using Softmax
        output_data = tf.nn.softmax(output_data, axis=1).numpy()

        # Determine the predicted speech command
        predicted_class = np.argmax(output_data[0])

        # Store the true label
        true_labels.append(label.numpy()[0])

        # Store the predicted label
        predicted_labels.append(predicted_class)

        # Store the prediction probabilities for ROC-AUC calculation
        prediction_scores.append(output_data[0])

        total_samples += 1

    # Stop the benchmark timer
    inference_time = time.perf_counter() - start_time

    # Stop CodeCarbon and calculate the carbon emissions
    emissions = tracker.stop()

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

    print("\n---------------------")
    print("Performance on the Keyword Spotting Benchmark")
    print(f"Accuracy: {accuracy:.4f}")

    MLPERF_TARGET = 0.90

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
    print(f"Carbon Emissions:     {emissions:.8f} kgCO2e")
    print("---------------------")

    # Print a summary of the inference performance
    print("Inference Summary")
    print(f"Inference Time:       {inference_time:.2f} seconds")
    print(f"Samples Processed:    {total_samples}")
    print(f"Throughput:           {throughput:.2f} samples/second")
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
    plt.show()

    print("Confusion matrix saved to:")
    print(confusion_matrix_path)
    print("---------------------")

    # Convert the true labels into a one-hot representation for multi-class ROC calculation
    y_true_bin = label_binarize(
        true_labels,
        classes=range(num_labels)
    )

    # Plot the One-vs-Rest ROC curve
    plt.figure(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_true_bin.ravel(),
        np.array(prediction_scores).ravel(),
        name="Micro-average ROC",
        color="darkorange"
    )

    plt.title("Keyword Spotting ROC Curve")
    plt.tight_layout()

    roc_curve_path = os.path.join(
        RESULTS_FOLDER,
        "roc_curve.png"
    )

    plt.savefig(roc_curve_path, dpi=300)
    plt.show()

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
        "keyword_spotting_predictions.csv"
    )

    results_df.to_csv(csv_path, index=False)

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    print("\n==========================================")
    print("Benchmark completed.")
    print("==========================================")
    print(f"Device:              {device_name}")
    print(f"Workload:            {workload_type}")
    print(f"Samples Evaluated:   {total_samples}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"Inference Time:      {inference_time:.2f} seconds")
    print(f"Throughput:          {throughput:.2f} samples/second")
    print(f"Carbon Emissions:    {emissions:.8f} kgCO2e")
    print("==========================================")