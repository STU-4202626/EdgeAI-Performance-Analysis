# EdgeAI - Performance Analysis of Edge Nodes for AI Tasks

**How much AI can you actually run on the edge?**

> [!IMPORTANT]
> The aim is to see how different devices actually perform when AI is moved away from the cloud and onto the edge.


EdgeAI is an Honours Computer Science project investigating how different edge devices handle AI workloads locally, from lightweight TinyML workloads to local large language model inference.

The project combines two benchmarking approaches:

- **MLPerf Tiny workloads** for image classification, keyword spotting and visual wake words, based on the MLPerf Tiny benchmark suite and adapted for this project.
- A **custom LLM benchmark** developed for this project using the lightweight **Gemma 3 1B (`gemma3:1b`)** model.

The project looks beyond inference speed alone by recording measurements such as **inference time, throughput, classification performance and estimated operational carbon emissions**. This allows the performance and resource requirements of the same AI workloads to be compared across very different types of hardware.

## Benchmarks

### MLPerf Tiny

The MLPerf Tiny part of the project uses workloads from the **MLPerf Tiny benchmark suite** as its foundation.

The completed implementation covers:

| Workload | Dataset | Model |
|---|---|---|
| Image Classification | CIFAR-10 | ResNet8 |
| Keyword Spotting | Speech Commands v0.02 | DS-CNN |
| Visual Wake Words | Visual Wake Words | MobileNet V1 |

The evaluation uses **200 CIFAR-10 images**, **1,000 Speech Commands samples**, and **1,000 Visual Wake Words images** (500 person and 500 non-person).


The original MLPerf Tiny scripts were adapted and extended for the requirements of this project. The modifications include changes to the benchmarking process, result collection, environmental measurements and result compilation.

The workloads were evaluated across **five platforms and six deployment configurations**. The Raspberry Pi 5 was tested using both full-precision TensorFlow and INT8 LiteRT. The CPU and GPU used full-precision TensorFlow, the Galaxy S26 Ultra used INT8 LiteRT, and the ESP32-S3 used INT8 TensorFlow Lite Micro.

### ESP32-S3 Implementation

The ESP32-S3 benchmarking was completed this term. Unlike the other platforms, the ESP32-S3 uses **Python host scripts together with firmware running on the microcontroller**. A separate firmware project was developed for each MLPerf Tiny workload.

| Component | Configuration |
|---|---|
| Board | Waveshare ESP32-S3, 16 MB flash and 8 MB Octal PSRAM |
| Processor | Dual-core Xtensa LX7, 240 MHz |
| Firmware framework | ESP-IDF 6.0.3 |
| Espressif inference components | `esp-tflite-micro` 1.4.0 (TensorFlow Lite Micro) and `esp-nn` 1.3.2 |
| Models | INT8-quantised ResNet8, DS-CNN and MobileNet V1 |
| Serial connection | 921600 baud |

The host prepares the dataset samples and sends **INT8 input tensors** over serial. The ESP32-S3 performs inference using TensorFlow Lite Micro, with the model embedded in flash and the tensor arena allocated in external PSRAM. The arena size depends on the workload: Image Classification and Keyword Spotting each reserve 128 KiB, while Visual Wake Words reserves 1 MiB. The device returns the quantised model output and device-side inference time. The host then calculates the classification measurements and saves the results.

The firmware uses `PING`/`PONG` to verify the connection and `INFR`/`RSLT` for inference requests and responses. **Only the time spent executing the model on the microcontroller is included in the reported ESP32-S3 inference time.** Host preprocessing and serial transfer are excluded. Reported throughput also excludes the firmware's 500 ms delay after each command.

> [!NOTE]
> Each workload has its own `firmware/` directory under `MLPerf_Tiny/MLPerf_Tiny_ESP32-S3/training/`. The repository includes the firmware source, embedded `model.cc`, `sdkconfig.defaults`, `partitions.csv`, component manifests and dependency lock files. Generated builds and downloaded `managed_components/` are excluded.

### Local LLM Benchmark

The LLM benchmark was developed separately for this project.

It uses: **Gemma 3 1B - `gemma3:1b`**

Only one LLM model is used so that the comparison focuses on differences between devices rather than differences between models.

The benchmark was designed around workload categories developed for this project after reviewing relevant research literature. It measures local inference performance as well as resource and environmental usage.

The benchmark evaluates **Knowledge Retrieval, Summarisation, Logical Reasoning, Mathematical Reasoning, and Code Generation** using the same fixed prompts on the four supported platforms. It measures inference performance, not the correctness or quality of the generated answers.

> [!IMPORTANT]
> The LLM benchmark uses **Gemma 3 1B (`gemma3:1b`)** as its single model. It ran on the Ryzen 7 6800H, RTX 4090, Raspberry Pi 5 and Galaxy S26 Ultra. **The ESP32-S3 is recorded as unsupported**, because the model's memory and storage requirements exceed those of the microcontroller.



## Hardware

The benchmarks were completed across devices with substantially different levels of computing capability.

| Device | Type |
|---|---|
| AMD Ryzen 7 6800H | x86 CPU |
| NVIDIA GeForce RTX 4090 | Dedicated GPU |
| Raspberry Pi 5 | ARM edge computer |
| Samsung Galaxy S26 Ultra | Android smartphone |
| ESP32-S3 | Microcontroller |

The ESP32-S3, Raspberry Pi 5 and Galaxy S26 Ultra are the three edge platforms. The Ryzen 7 6800H and RTX 4090 serve as higher-performance references; the RTX 4090 evaluation was conducted on RunPod.



## What is being measured?

### MLPerf Tiny

The MLPerf Tiny workloads record measurements including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Inference time (s)
- Throughput (images/s for Image Classification and Visual Wake Words; samples/s for Keyword Spotting)
- Estimated operational carbon emissions (kgCO2e)

The benchmark results also record the relevant MLPerf target and whether the benchmark passed that target.

> [!NOTE]
> MLPerf Tiny inference time refers to **one complete workload**, not one individual sample. On the CPU, GPU, Raspberry Pi 5 and Android, it is elapsed prediction-loop time per repetition, including runtime and output handling. On the ESP32-S3, it is summed device-side `Invoke()` time; these timing scopes differ. In the compiled CSV, accuracy and classification scores are stored as fractions (for example, `0.875`); the accuracy results shown below are expressed as percentages (87.5%).

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
- Estimated operational carbon emissions (kgCO2e)

The available measurements can vary depending on the device and the tools available on that platform. Each benchmark included **one warm-up run followed by three measured runs**. For MLPerf Tiny, each measured run repeated the complete workload until at least 30 seconds of elapsed prediction-loop time was reached on the CPU, GPU, Raspberry Pi 5 and Android. The ESP32-S3 instead accumulated at least 30 seconds of device-side model inference time. Each run completed the current workload repetition before stopping.

> [!NOTE]
> CodeCarbon provides software-based estimates on the CPU, GPU, Raspberry Pi 5 and Android platforms. For the ESP32-S3, energy and carbon emissions were estimated from device-side inference time, a datasheet-based power value of approximately **0.290 W**, and a carbon-intensity factor of **0.707686 kgCO2e/kWh**. These figures are **not direct whole-device electricity measurements**. CodeCarbon logs use different regional carbon intensities and power estimates, including a 42.5 W CPU fallback on Android, so emissions comparisons also reflect these estimation assumptions.

## Repository Structure

```text
EdgeAI-Performance-Analysis/
├── LLM/
│   ├── LLM_Bench_Android/
│   ├── LLM_Bench_PC_CPU/
│   ├── LLM_Bench_PC_GPU/
│   ├── LLM_Bench_Pi5/
│   ├── compile_results.py
│   └── LLM_Bench_Results.csv
├── MLPerf_Tiny/
│   ├── MLPerf_Tiny_Android/
│   ├── MLPerf_Tiny_PC_CPU/
│   ├── MLPerf_Tiny_PC_GPU/
│   ├── MLPerf_Tiny_RaspberryPi5/
│   ├── MLPerf_Tiny_ESP32-S3/
│   │   ├── mlperf_tiny_ESP32S3_summary.csv
│   │   └── training/
│   │       ├── esp32_energy_helper.py
│   │       ├── image_classification/  (host script, firmware/, results/)
│   │       ├── keyword_spotting/      (host script, firmware/, results/)
│   │       └── visual_wake_words/    (host script, firmware/, results/)
│   ├── compile_results.py
│   ├── compile_all_devices.py
│   └── mlperf_tiny_all_devices_summary.csv
├── .gitignore
└── README.md
```

Each `MLPerf_Tiny_*` platform folder contains a `training/` directory with the three workload implementations and their recorded results. The ESP32-S3 additionally includes the firmware projects and a device-specific energy helper.

Large benchmark datasets and generated ESP-IDF dependencies are excluded from the current repository tree. Dataset sources are listed below.

## Datasets

The MLPerf Tiny workloads require:

| Dataset | Used for | Source |
|---|---|---|
| **CIFAR-10** | Image Classification | [University of Toronto - CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **Speech Commands v0.02** | Keyword Spotting | [TensorFlow - Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) |
| **Visual Wake Words** | Visual Wake Words | [MLCommons - MLPerf Tiny](https://github.com/mlcommons/tiny) |


The benchmark directories contain the scripts and configuration required to work with these datasets.

> [!IMPORTANT]
> The benchmark datasets are **not included in the current repository tree** because of their size. You will need to download the required datasets separately before running the benchmarks. Exported Keyword Spotting test tensors remain in earlier Git history; their local copies have been retained after removal from tracking.

### Dataset notes

**CIFAR-10**

CIFAR-10 is a small image classification dataset containing **60,000 colour images**, each with a resolution of **32 × 32 pixels**. The images are divided into **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. The dataset contains 50,000 training images and 10,000 test images. [University of Toronto - CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

For this project, CIFAR-10 is used by the **Image Classification** benchmark with the ResNet8 model.

**Speech Commands v0.02**

Speech Commands is an audio dataset designed for **keyword spotting**. It contains more than **105,000 WAV audio recordings** covering spoken words from **35 different word categories**, recorded from multiple speakers. [TensorFlow - Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands)

For this project, the **Speech Commands v0.02** dataset is used by the **Keyword Spotting** benchmark with the DS-CNN reference model. The evaluation uses 12 output classes, rather than treating all 35 recorded word categories as separate benchmark classes.

**Visual Wake Words**

Visual Wake Words is an image classification dataset designed for **person detection** on resource-constrained devices. It is derived from the **COCO dataset** and converts the original object-detection information into a binary classification task: **person** or **non-person**. Images are processed for the benchmark at **96 × 96 pixels**. [MLCommons - MLPerf Tiny](https://github.com/mlcommons/tiny)

For this project, Visual Wake Words is used by the **Visual Wake Words** benchmark with the MobileNet V1 reference model.

The exact dataset preparation and directory structure used by the benchmarks can be found in the relevant `MLPerf_Tiny` workload directories.

## Results

The compiled MLPerf Tiny results are available in:

```text
MLPerf_Tiny/mlperf_tiny_all_devices_summary.csv
```

The ESP32-S3 platform summary is available in `MLPerf_Tiny/MLPerf_Tiny_ESP32-S3/mlperf_tiny_ESP32S3_summary.csv`.

The compiled LLM results are available in:

```text
LLM/LLM_Bench_Results.csv
```

The repository also contains individual benchmark result CSVs, predictions and selected figures used to examine each workload. The final compiled results include **18 MLPerf Tiny workload-configuration records** and **20 local LLM workload records**. The Android and INT8 Raspberry Pi 5 Keyword Spotting and Visual Wake Words results accompany the historical scripts from `7b0ae71`; later local fixes have not been benchmarked. See the [result provenance and CSV precision notes](MLPerf_Tiny/RESULTS_PROVENANCE.md).

### ESP32-S3 Results

The ESP32-S3 completed all three MLPerf Tiny workloads using INT8 models:

| Workload | Accuracy | Required accuracy | Throughput |
|---|---:|---:|---:|
| Image Classification | 87.5% | 85% | 11.24 images/s |
| Keyword Spotting | 91.2% | 90% | 50.45 samples/s |
| Visual Wake Words | 85.8% | 80% | 12.95 images/s |

> [!IMPORTANT]
> **All three ESP32-S3 workloads met their MLPerf Tiny accuracy targets.** The results demonstrate that these lightweight models can run on the microcontroller, even though its inference throughput is lower than on the more capable INT8 platforms.

### Key Findings

- The **Samsung Galaxy S26 Ultra** achieved the highest MLPerf Tiny throughput using INT8 LiteRT, while the **RTX 4090** led the full-precision configurations and local Gemma 3 1B inference.
- On the **Raspberry Pi 5**, changing from full-precision TensorFlow to INT8 LiteRT increased throughput by **67.81×** for Image Classification, **3.68×** for Keyword Spotting and **approximately 70.7x** for Visual Wake Words. Both model representation and inference runtime changed, so this cannot be attributed to quantisation alone.
- The **ESP32-S3** met all three TinyML accuracy targets but could not support Gemma 3 1B. The results show why the model and runtime must be considered alongside available memory.
- **Estimated operational emissions depended on the workload.** The RTX 4090 had higher estimates than the Ryzen 7 6800H on the three full-precision MLPerf Tiny workloads, but lower estimates on all five Gemma 3 1B prompts. The ESP32-S3 estimates exceeded those of the INT8 Raspberry Pi 5 despite its lower assumed power because its inference times were longer. These are not direct measurements of whole-device energy consumption.

## Running the Benchmarks

The benchmark implementations are complete. The general execution process is:

1. Prepare the required dataset and model.
2. Set up the required environment and dependencies.
3. Run the device-specific benchmark.
4. Save the benchmark results and environmental measurements.
5. Compile the results for comparison across devices.

Device-specific benchmark scripts are located inside their respective directories. The required datasets must be downloaded separately before running them.

### ESP32-S3 Firmware

The three firmware projects are located at `MLPerf_Tiny/MLPerf_Tiny_ESP32-S3/training/<workload>/firmware/`, where `<workload>` is `image_classification`, `keyword_spotting` or `visual_wake_words`. Workload-specific instructions are available for [Image Classification](MLPerf_Tiny/MLPerf_Tiny_ESP32-S3/training/image_classification/firmware/README.md), [Keyword Spotting](MLPerf_Tiny/MLPerf_Tiny_ESP32-S3/training/keyword_spotting/firmware/README.md) and [Visual Wake Words](MLPerf_Tiny/MLPerf_Tiny_ESP32-S3/training/visual_wake_words/firmware/README.md).

1. Use the **ESP-IDF 6.0.3** environment and select the appropriate workload's `firmware/` directory.
2. Build and flash the relevant firmware to the ESP32-S3. ESP-IDF restores the required Espressif components from `idf_component.yml` and `dependencies.lock`.
3. From the matching workload directory, set the host script's `SERIAL_PORT` (currently `COM8`) to the board's UART port, close any serial monitor, and run the corresponding `mlperf_*_ESP32S3_bench.py` host script with the matching dataset available at **921600 baud**.
4. The host script records the inference outputs and writes the workload result CSVs; compilation scripts then produce the platform and cross-platform summaries.

The tracked `sdkconfig.defaults` files preserve the **16 MB flash**, **80 MHz Octal PSRAM**, and **240 MHz CPU** configuration. The active `sdkconfig`, `build/` and `managed_components/` files are intentionally excluded as generated or local build state.

> [!NOTE]
> Firmware build and host execution depend on the local ESP-IDF, serial-port and dataset setup. A clean build from the tracked files has not been validated in this review. The cross-platform results in this repository were recorded under the configurations described above; they are **not official MLPerf Tiny submissions**.

## MLPerf Tiny

This project uses and adapts components of the **MLPerf Tiny** benchmark suite developed by **MLCommons**.

The original MLPerf Tiny benchmark suite, scripts and associated materials remain the work of their respective authors and are subject to their original licensing and attribution requirements.

EdgeAI does not claim ownership of the original MLPerf Tiny benchmark suite.

The MLPerf Tiny scripts included in this repository have been modified for the requirements of this project. These modifications include changes to the benchmarking process, result collection, environmental measurements and result compilation.

The original MLPerf Tiny implementation was also used as a reference when developing parts of the benchmarking workflow.

The complete, unchanged [MLPerf Tiny Apache 2.0 licence](MLPerf_Tiny/LICENSE.md) is included alongside the [source attribution and modification notes](MLPerf_Tiny/ATTRIBUTION.md). Original notices are retained, and adapted upstream source files carry modification notices.

> [!NOTE]
> EdgeAI does not reproduce the MLPerf Tiny suite unchanged. The original workloads and implementation were used as a foundation and reference, with the relevant scripts adapted for this project.

## Project Status

### Completed

- [x] MLPerf Tiny benchmark implementation
- [x] MLPerf Tiny result compilation
- [x] Custom LLM benchmark
- [x] Gemma 3 1B local inference benchmarking
- [x] Estimated operational carbon-emissions results, including the ESP32-S3 energy calculation
- [x] Cross-device result compilation
- [x] Benchmark result collection across multiple devices
- [x] ESP32-S3 Image Classification, Keyword Spotting and Visual Wake Words firmware and host benchmarks
- [x] ESP32-S3 result compilation and accuracy-target verification
- [x] Benchmark visualisations and final comparative analysis
- [x] Final Honours report and conference-paper draft

### In Progress

- [ ] Final presentation and demonstration recordings
- [ ] Final project website and repository publishing

## Author

**STU-4202626**

BSc Honours Computer Science\
University of the Western Cape
