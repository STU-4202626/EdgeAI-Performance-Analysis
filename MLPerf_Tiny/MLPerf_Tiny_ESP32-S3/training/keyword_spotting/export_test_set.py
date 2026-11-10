"""
MLCommons Tiny Benchmark
Keyword Spotting (Speech Commands)

Benchmark Dataset Exporter
------------------------
Exports the official MLPerf Tiny Speech Commands test set to NumPy arrays.

The Android benchmark uses TensorFlow Lite Runtime instead of TensorFlow,
so the benchmark features are generated once on the PC using the official
MLPerf Tiny preprocessing pipeline and then reused on Android. This ensures
that every device is evaluated using exactly the same benchmark samples while
avoiding the need to run TensorFlow on Android.
"""

import os
import numpy as np
import get_dataset as kws_data
import kws_util

# Number of Speech Commands samples to export
NUM_SAMPLES = 1000

# Folder used to store the exported benchmark dataset
BENCHMARK_FOLDER = "benchmark_data"

os.makedirs(BENCHMARK_FOLDER, exist_ok=True)

# Load the MLPerf Tiny benchmark configuration
Flags, _ = kws_util.parse_command()

# Load the official Speech Commands dataset
_, ds_test, _ = kws_data.get_training_data(Flags)

# Select the benchmark test samples that will be used on Android
dataset = ds_test.unbatch().take(NUM_SAMPLES).batch(1)

# Lists used to store the benchmark features and labels
features = []
labels = []

print("Exporting benchmark test set...\n")

# Extract each benchmark sample and store it as a NumPy array
for feature, label in dataset:
    features.append(feature.numpy())
    labels.append(label.numpy())

# Combine all samples into a single array
x_test = np.concatenate(features)

# Combine all labels into a single array.
y_test = np.concatenate(labels)

# Save the benchmark features.
np.save(os.path.join(BENCHMARK_FOLDER, "x_test.npy"), x_test)

# Save the benchmark labels.
np.save(os.path.join(BENCHMARK_FOLDER, "y_test.npy"), y_test)

print("Benchmark dataset exported successfully.\n")

print(f"Feature Shape: {x_test.shape}")
print(f"Label Shape:   {y_test.shape}")