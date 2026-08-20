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
import vww_model_updated
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, precision_score, recall_score, f1_score)
from codecarbon import EmissionsTracker



# Device and workload information
device_name = "Samsung_Galaxy_S26_Ultra"
workload_type = "Visual Wake Words"

# Dataset configuration
IMAGE_SIZE = (96, 96)
NUM_IMAGES = 1000

# Root directory containing the Visual Wake Words dataset
DATASET_DIR = "vw_coco2014_96"

# Individual class folders
PERSON_DIR = os.path.join(DATASET_DIR, "person")
NON_PERSON_DIR = os.path.join(DATASET_DIR, "non_person")

# Visual Wake Words dataset location
dataset_path = "vw_coco2014_96"

# Folder for saving benchmark results
results_folder = "results"

os.makedirs(results_folder, exist_ok=True)

if __name__ == "__main__":

    # Load the pre-trained MobileNet V1 model used for the MLPerf Tiny Visual Wake Words benchmark
    model = vww_model_updated.mobilenet_v1()

    model.load_weights("trained_models/vww_96.h5")

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

    print("Dataset Information")
    print("---------------------")
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

    # Create lists to store the true labels, predicted labels, and prediction probabilities
    predicted_labels = []
    prediction_probabilities = []

    # Monitor power consumption and carbon emissions during benchmark execution
    tracker = EmissionsTracker(
        project_name=f"{device_name}_{workload_type}",
        measure_power_secs=1,
        log_level="error",
        gpu_ids=[],
        output_dir=results_folder,
        output_file="emissions.csv"
    )

    # Start monitoring benchmark execution time and carbon emissions
    tracker.start()

    start_time = time.perf_counter()

    # Run inference on each image in the balanced evaluation dataset
    for image_path in image_paths:

        # Load and resize the image to 96x96 pixels
        img = image.load_img(image_path, target_size=IMAGE_SIZE)

        # Convert the image into a NumPy array
        img_array = image.img_to_array(img)

        # Normalise pixel values to the range [0,1]
        img_array = img_array / 255.0

        # Add a batch dimension expected by TensorFlow
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        prediction = model.predict(img_array, verbose=0)

        # Store the predicted class
        predicted_labels.append(np.argmax(prediction))

        # Store the probability of the "person" class
        prediction_probabilities.append(prediction[0][1])

    # Stop monitoring benchmark execution time and carbon emissions
    end_time = time.perf_counter()

    emissions = tracker.stop()

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

    print("\n---------------------")
    print("Performance on the Visual Wake Words Benchmark")
    print(f"Accuracy: {accuracy:.4f}")

    MLPERF_TARGET = 0.80

    print(f"MLPerf Target:       {MLPERF_TARGET:.0%}")

    if accuracy >= MLPERF_TARGET:
        print("Benchmark Status:    PASSED")
    else:
        print("Benchmark Status:    FAILED")

    print("---------------------")

    # Generate a detailed classification report containing:
    # Precision, Recall, F1-Score and Support for each class
    print("Classification Report")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    ))
    print("---------------------")

    # Print out power emissions
    print("Emission Report")
    print(f"Carbon Emissions:     {emissions:.8f} kgCO2e")
    print("---------------------")

    # Calculate total benchmark execution time
    total_time = end_time - start_time

    # Calculate average throughput (images processed per second)
    throughput = len(true_labels) / total_time

    print("Inference Summary")
    print(f"Inference Time:       {total_time:.2f} seconds")
    print(f"Images Processed:     {len(true_labels)}")
    print(f"Throughput:           {throughput:.2f} images/second")
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

    plt.show()

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

    plt.show()

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
        "visual_wake_words_predictions.csv"
    )

    results_df.to_csv(csv_path, index=False)

    print("Prediction results saved to:")
    print(csv_path)
    print("---------------------")

    print()
    print("==========================================")
    print("Benchmark completed.")
    print("==========================================")
    print(f"Device:              {device_name}")
    print(f"Workload:            {workload_type}")
    print(f"Images Evaluated:    {len(true_labels)}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"Inference Time:      {total_time:.2f} seconds")
    print(f"Throughput:          {throughput:.2f} images/second")
    print(f"Carbon Emissions:    {emissions:.8f} kgCO2e")
    print("==========================================")