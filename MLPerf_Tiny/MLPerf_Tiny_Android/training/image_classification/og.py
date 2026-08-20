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
import numpy as np
import matplotlib.pyplot as plt
import pickle
import train
import eval_functions_eembc
from codecarbon import EmissionsTracker
from sklearn.metrics import (roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,)
from tflite_runtime.interpreter import Interpreter

# if True uses the official MLPerf Tiny subset of CIFAR10 for validation
# if False uses the full CIFAR10 validation set
PERF_SAMPLE = True

MODEL_PATH = "trained_models/pretrainedResnet_quant.tflite"

# Device information
device_name = "Samsung_Galaxy_S26_Ultra"
workload_type = "Image Classification"

# Folder for benchmark outputs
results_folder = "results"

os.makedirs(results_folder, exist_ok=True)

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

    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Number of labels:", len(label_names))
    print("Label names: ", label_names)
    label_classes = np.argmax(test_labels,axis=1)
    print("Label classes: ", label_classes.shape)

    # Load the pre-trained MLPerf Tiny ResNet model used for CIFAR-10 image classification
    interpreter = Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input Tensor:",
      input_details[0]["shape"],
      input_details[0]["dtype"])

    print("Output Tensor:",
        output_details[0]["shape"],
        output_details[0]["dtype"])

    device_name = "Samsung_Galaxy_S26_Ultra"
    workload_type = "Image Classification"

    # Calc Power Emissions - SeanB
    tracker = EmissionsTracker(
        project_name=f"{device_name}_{workload_type}",
        measure_power_secs=1,
        log_level="error",
        gpu_ids=[],
        output_dir=results_folder,
        output_file="emissions.csv"
    )

    tracker.start()

    # Record the start time before inference begins in order to calculate the total inference time
    start_time = time.perf_counter()

    predictions = []

    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    for image in test_data:

        input_data = np.expand_dims(image, axis=0)

        if input_details[0]["dtype"] == np.int8:

            input_data = np.round(
                input_data / input_scale + input_zero_point
            ).astype(np.int8)

        else:

            input_data = input_data.astype(np.float32)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        if output_details[0]["dtype"] == np.int8:

            output = output_scale * (
                output.astype(np.float32) - output_zero_point
            )

        predictions.append(output[0])

    predictions = np.array(predictions)

    # Calculate the total inference time required to classify the benchmark dataset
    inference_time = time.perf_counter() - start_time

    emission = tracker.stop()

    # Convert probability predictions into the most likely class for each image using the highest probability score
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate classification accuracy using scikit-learn
    # Accuracy = Correct Predictions / Total Predictions
    accuracy = accuracy_score(label_classes, predicted_classes)

    # Calculate additional performance metrics used to evaluate the benchmark
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

    # Calculate the number of benchmark images processed during inference
    samples_processed = len(label_classes)

    # Calculate the average number of images processed per second
    throughput = samples_processed / inference_time

    print("\n---------------------")
    print("Performances on cifar10 test set")
    print("Accuracy:", accuracy)

    MLPERF_TARGET = 0.85

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

    plt.show()

    # Print out power emission - SeanB
    print("Emission Report")
    print(f"Carbon Emissions:     {emission:.8f} kgCO2e\n")
    print("---------------------")

    # Calculate accuracy using the official EEMBC evaluation function to compare against the scikit-learn result
    print("EEMBC calculate_accuracy method")
    accuracy_eembc = eval_functions_eembc.calculate_accuracy(predictions, label_classes)
    print("---------------------")

    # Compute multi-class ROC-AUC using prediction probabilities.
    # ROC-AUC evaluates the model's ability to distinguish between different CIFAR-10 classes.
    auc_scikit = roc_auc_score(test_labels, predictions)
    print("sklearn.metrics.roc_auc_score method")
    print("AUC sklearn: ", auc_scikit)
    print("---------------------")

    # Compute ROC-AUC using the official EEMBC evaluation implementation
    print("EEMBC calculate_auc method")
    auc_eembc = eval_functions_eembc.calculate_auc(predictions, label_classes, label_names, model_name)
    print("---------------------")

    # Save the prediction results for every benchmark image to a CSV file for further analysis
    results_df = pd.DataFrame({
        "True Label": label_classes,
        "Predicted Label": predicted_classes,
        "Prediction Correct": label_classes == predicted_classes
    })

    csv_path = os.path.join(
        results_folder,
        "image_classification_predictions.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    # Display a summary of the inference performance measured during benchmarking
    print("Inference Summary")
    print(f"Inference Time:       {inference_time:.2f} seconds")
    print(f"Images Processed:     {samples_processed}")
    print(f"Throughput:           {throughput:.2f} images/second")
    print("---------------------")

    # Display the final benchmark results for the evaluated device
    print("\n==========================================")
    print("Benchmark completed.")
    print("==========================================")
    print(f"Device:              {device_name}")
    print(f"Workload:            {workload_type}")
    print(f"Images Evaluated:    {samples_processed}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {auc_scikit:.4f}")
    print(f"Inference Time:      {inference_time:.2f} seconds")
    print(f"Throughput:          {throughput:.2f} images/second")
    print(f"Carbon Emissions:    {emission:.8f} kgCO2e")
    print("==========================================")