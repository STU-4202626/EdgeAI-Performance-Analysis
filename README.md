# EdgeAI - Performance Analysis of Edge Nodes for AI Tasks

**How much AI can you actually run on the edge?**

> **The aim is to see how different devices actually perform when AI is moved away from the cloud and onto the edge.**

EdgeAI is an Honours Computer Science project investigating how different edge devices handle AI workloads locally, from lightweight TinyML workloads to local large language model inference.

The project combines two benchmarking approaches:

- **MLPerf Tiny workloads** for image classification, keyword spotting and visual wake words, based on the MLPerf Tiny benchmark suite and adapted for this project.
- A **custom LLM benchmark** developed for this project using the lightweight **Gemma 3 1B (`gemma3:1b`)** model.

The project looks beyond inference speed alone by recording measurements such as **inference time, throughput, energy consumption and carbon emissions**. This allows the performance and resource requirements of the same AI workloads to be compared across very different types of hardware.

## Benchmarks

### MLPerf Tiny

The MLPerf Tiny part of the project uses workloads from the **MLPerf Tiny benchmark suite** as its foundation.

The current implementation covers:

| Workload | Dataset | Model |
|---|---|---|
| Image Classification | CIFAR-10 | ResNet |
| Keyword Spotting | Speech Commands v0.02 | KWS reference model |
| Visual Wake Words | Visual Wake Words | VWW |


The original MLPerf Tiny scripts were adapted and extended for the requirements of this project. The modifications include changes to the benchmarking process, result collection, environmental measurements and result compilation.

The workloads are run across multiple devices so that their performance can be compared under the same general benchmark conditions.

### Local LLM Benchmark

The LLM benchmark was developed separately for this project.

It uses: **Gemma 3 1B - `gemma3:1b`**

Only one LLM model is used so that the comparison focuses on differences between devices rather than differences between models.

The benchmark was designed around workload categories developed for this project after reviewing relevant research literature. It measures local inference performance as well as resource and environmental usage.

The current LLM benchmark records:

- Time to First Token (TTFT)
- Tokens Per Second (TPS)
- CPU usage
- Memory usage
- Energy consumption
- Carbon emissions

The benchmark is intended to provide a consistent way of comparing local LLM inference across the selected edge devices.

> [!IMPORTANT]
> The LLM benchmark uses **Gemma 3 1B (`gemma3:1b`)** as its single model. It was selected because its relatively small size makes local inference more practical on resource-constrained edge devices.



## Hardware

The benchmarks are being run across devices with substantially different levels of computing capability.

| Device | Type |
|---|---|
| AMD Ryzen 7 6800H | x86 CPU |
| NVIDIA GeForce RTX 4090 | Dedicated GPU |
| Raspberry Pi 5 | ARM edge computer |
| Samsung Galaxy S26 Ultra | Android smartphone |
| ESP32-S3 | Microcontroller

The project focuses on comparing how the same or equivalent workloads behave across these different platforms.



## What is being measured?

### MLPerf Tiny

The MLPerf Tiny workloads record measurements including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Inference time
- Throughput
- Energy consumption
- Carbon emissions

The benchmark results also record the relevant MLPerf target and whether the benchmark passed that target.

### LLM

The LLM benchmark focuses on:

- Total Duration (s)
- Load Duration (s)
- Prompt Evaluation Count
- Prompt Evaluation Duration (s)
- Prompt Evaluation Rate (tokens/s)
- Evaluation Count
- Evaluation Duration (s)
- Evaluation Rate (tokens/s)
- Energy consumption
- Carbon emissions

The available measurements can vary depending on the device and the tools available on that platform.

## Repository Structure

```
EdgeAI-Performance-Analysis/
├── LLM
│   ├── LLM_Bench_Android
│   ├── LLM_Bench_PC_CPU
│   ├── LLM_Bench_PC_GPU
│   ├── LLM_Bench_Pi5
│   ├── compile_results.py
│   └── LLM_Bench_Results.csv
├── MLPerf_Tiny
│   ├── MLPerf_Tiny_Android
│   │   ├── image_classification
│   │   ├── keyword_spotting
│   │   └── visual_wake_words
│   ├── MLPerf_Tiny_PC_CPU
│   │   ├── image_classification
│   │   ├── keyword_spotting
│   │   └── visual_wake_words
│   ├── MLPerf_Tiny_PC_GPU
│   │   ├── image_classification
│   │   ├── keyword_spotting
│   │   └── visual_wake_words
│   ├── MLPerf_Tiny_RaspberryPi5
│   │   ├── image_classification
│   │   ├── keyword_spotting
│   │   └── visual_wake_words
│   ├── compile_results.py
│   ├── compile_all_devices.py
│   └── mlperf_tiny_all_devices_summary.csv
├── .gitignore
└── README.md
```

The repository contains the benchmark implementations, result files and scripts used to process and compile results across devices.

Large datasets used by the benchmarks are intentionally not included in the repository because of their size. The required datasets and their download sources will be documented below.

## Datasets

The current MLPerf Tiny workloads require:

| Dataset | Used for | Source |
|---|---|---|
| **CIFAR-10** | Image Classification | [University of Toronto - CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **Speech Commands v0.02** | Keyword Spotting | [TensorFlow - Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) |
| **Visual Wake Words** | Visual Wake Words | [MLCommons - MLPerf Tiny](https://github.com/mlcommons/tiny) |


The benchmark directories contain the scripts and configuration required to work with these datasets.

> [!IMPORTANT]
> The benchmark datasets are **not included in this repository** because of their size. You will need to download the required datasets separately before running the benchmarks.

### Dataset notes

**CIFAR-10**

CIFAR-10 is a small image classification dataset containing **60,000 colour images**, each with a resolution of **32 × 32 pixels**. The images are divided into **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. The dataset contains 50,000 training images and 10,000 test images. [University of Toronto - CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

For this project, CIFAR-10 is used by the **Image Classification** benchmark with the ResNet model.

**Speech Commands v0.02**

Speech Commands is an audio dataset designed for **keyword spotting**. It contains more than **105,000 WAV audio recordings** covering spoken words from **35 different word categories**, recorded from multiple speakers. [TensorFlow - Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands)

For this project, the **Speech Commands v0.02** dataset is used by the **Keyword Spotting** benchmark with the KWS reference model.

**Visual Wake Words**

Visual Wake Words is an image classification dataset designed for **person detection** on resource-constrained devices. It is derived from the **COCO dataset** and converts the original object-detection information into a binary classification task: **person** or **non-person**. Images are processed for the benchmark at **96 × 96 pixels**. [MLCommons - MLPerf Tiny](https://github.com/mlcommons/tiny)

For this project, Visual Wake Words is used by the **Visual Wake Words** benchmark with the VWW model.

The exact dataset preparation and directory structure used by the benchmarks can be found in the relevant `MLPerf_Tiny` workload directories.

## Results

The compiled MLPerf Tiny results are located under:

```text
MLPerf_Tiny/
```

The compiled LLM results are available in:

```text
LLM/LLM_Bench_Results.csv
```

The repository also contains individual benchmark result files and the environmental measurements generated during the experiments.

Benchmark predictions, logs and other supporting result files are retained where they are useful for examining the individual runs.

Additional visualisations will be added as the project analysis progresses.

## Running the Benchmarks

Detailed setup and execution instructions will be added as the benchmark implementations are finalised.

In general, the benchmark process follows these steps:

1. Prepare the required dataset and model.
2. Set up the required environment and dependencies.
3. Run the device-specific benchmark.
4. Save the benchmark results and environmental measurements.
5. Compile the results for comparison across devices.

Device-specific benchmark scripts are located inside their respective directories.

## MLPerf Tiny

This project uses and adapts components of the **MLPerf Tiny** benchmark suite developed by **MLCommons**.

The original MLPerf Tiny benchmark suite, scripts and associated materials remain the work of their respective authors and are subject to their original licensing and attribution requirements.

EdgeAI does not claim ownership of the original MLPerf Tiny benchmark suite.

The MLPerf Tiny scripts included in this repository have been modified for the requirements of this project. These modifications include changes to the benchmarking process, result collection, environmental measurements and result compilation.

The original MLPerf Tiny implementation was also used as a reference when developing parts of the benchmarking workflow.

Appropriate attribution and licensing information for the original MLPerf Tiny materials will be maintained alongside the relevant source files.

> [!NOTE]
> EdgeAI does not reproduce the MLPerf Tiny suite unchanged. The original workloads and implementation were used as a foundation and reference, with the relevant scripts adapted for this project.

## Project Status

### Completed

- [x] MLPerf Tiny benchmark implementation
- [x] MLPerf Tiny result compilation
- [x] Custom LLM benchmark
- [x] Gemma 3 1B local inference benchmarking
- [x] Carbon and energy measurements
- [x] Cross-device result compilation
- [x] Benchmark result collection across multiple devices


### In Progress

- [ ] Addition of ESP32-S3 as a benchmark device
- [ ] Benchmark visualisations
- [ ] Final comparative analysis
- [ ] Final project documentation

## Author

**STU-4202626**

Honors Computer Science  
University of the Western Cape