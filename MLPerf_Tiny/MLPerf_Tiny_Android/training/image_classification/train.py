"""
MLCommons TinyMLPerf

Android version of train.py

This file is only responsible for loading the CIFAR-10 dataset.
All TensorFlow/Keras training code has been removed because the Android benchmark only performs inference using a TFLite model.
"""

import numpy as np
import pickle


# Convert integer labels into one-hot encoded vectors.
# This replaces TensorFlow's to_categorical().
def to_categorical(labels, num_classes=10):

    labels = np.asarray(labels)

    return np.eye(num_classes, dtype=np.float32)[labels]


# Load one of the CIFAR-10 batch files.
def unpickle(file):

    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")

    return data


# Load the CIFAR-10 training and test datasets.
def load_cifar_10_data(data_dir, negatives=False):

    # Read the dataset metadata.
    meta_data_dict = unpickle(data_dir + "/batches.meta")

    cifar_label_names = np.array(
        meta_data_dict[b"label_names"]
    )

    # Training dataset
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):

        cifar_train_data_dict = unpickle(
            data_dir + f"/data_batch_{i}"
        )

        if i == 1:

            cifar_train_data = cifar_train_data_dict[b"data"]

        else:

            cifar_train_data = np.vstack(
                (
                    cifar_train_data,
                    cifar_train_data_dict[b"data"]
                )
            )

        cifar_train_filenames += (
            cifar_train_data_dict[b"filenames"]
        )

        cifar_train_labels += (
            cifar_train_data_dict[b"labels"]
        )

    cifar_train_data = cifar_train_data.reshape(
        (len(cifar_train_data), 3, 32, 32)
    )

    if negatives:

        cifar_train_data = (
            cifar_train_data
            .transpose(0, 2, 3, 1)
            .astype(np.float32)
        )

    else:

        cifar_train_data = np.rollaxis(
            cifar_train_data,
            1,
            4
        )

    cifar_train_filenames = np.array(
        cifar_train_filenames
    )

    cifar_train_labels = np.array(
        cifar_train_labels
    )

    # Test dataset
    cifar_test_data_dict = unpickle(
        data_dir + "/test_batch"
    )

    cifar_test_data = cifar_test_data_dict[b"data"]

    cifar_test_filenames = (
        cifar_test_data_dict[b"filenames"]
    )

    cifar_test_labels = (
        cifar_test_data_dict[b"labels"]
    )

    cifar_test_data = cifar_test_data.reshape(
        (len(cifar_test_data), 3, 32, 32)
    )

    if negatives:

        cifar_test_data = (
            cifar_test_data
            .transpose(0, 2, 3, 1)
            .astype(np.float32)
        )

    else:

        cifar_test_data = np.rollaxis(
            cifar_test_data,
            1,
            4
        )

    cifar_test_filenames = np.array(
        cifar_test_filenames
    )

    cifar_test_labels = np.array(
        cifar_test_labels
    )

    return (
        cifar_train_data,
        cifar_train_filenames,
        to_categorical(cifar_train_labels),
        cifar_test_data,
        cifar_test_filenames,
        to_categorical(cifar_test_labels),
        cifar_label_names,
    )