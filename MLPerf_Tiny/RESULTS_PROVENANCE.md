# Recorded Results and Script Provenance

This note preserves the distinction between the historical experiments and later local code corrections. No benchmarks were rerun and no recorded workload, platform-summary or LLM result files were recalculated during repository preparation.

## Android and Raspberry Pi KWS/VWW

The following four benchmark scripts have local fixes that are intentionally excluded from the prepared commit:

- `MLPerf_Tiny_Android/training/keyword_spotting/mlperf_keyword_spotting_Android_bench.py`
- `MLPerf_Tiny_Android/training/visual_wake_words/mlperf_visual_wake_words_Android_bench.py`
- `MLPerf_Tiny_RaspberryPi5/training/keyword_spotting/mlperf_keyword_spotting_Pi5_quant.py`
- `MLPerf_Tiny_RaspberryPi5/training/visual_wake_words/mlperf_visual_wake_words_Pi5_quant.py`

Git commit `7b0ae71` added the original scripts and their recorded results on **21 August 2026 at 01:54:58 SAST**. These committed versions convert KWS inputs directly to INT8 without clipping and apply an additional Softmax to the model output. The models already contain a Softmax output layer.

The corresponding CodeCarbon logs contain collection timestamps on **19 August 2026**; the log timestamps do not encode a timezone. The saved benchmark CSVs, prediction CSVs, benchmark logs and emissions logs match that commit after line-ending normalization. Local filesystem modification times place the four fixes on **23 September 2026**, and retained VWW copies under the ignored `Previous/` directory match the committed versions. Filesystem times are mutable supporting evidence, not proof of execution.

Together, Git history, retained outputs and local copies support treating these edits as post-collection fixes. The prepared snapshot retains the historical script logic alongside the historical results. Matching CSV values alone do not establish executable provenance, and there is no recorded evidence that the corrected working copies produced these results.

### Effect of the local fixes

The KWS fixes clip quantised inputs before INT8 conversion. Static inspection of the retained Android and Pi test tensors found 16 values in sample 563 (zero-based index) that quantise above 127, reaching 133 with input scale `0.5847029089927673` and zero point `83`. The historical conversion wraps these values; the corrected conversion saturates them at 127. This changes the evaluated input tensor and can affect predictions.

All four fixes retain dequantisation and replace the second Softmax with probability renormalisation. The change can affect prediction scores and ROC-AUC. Output handling also occurs inside the timed prediction loop, so timing, repetition counts and estimated emissions cannot be treated as measurements of the corrected code. The saved prediction CSVs contain labels and correctness, without probability vectors; they cannot independently reconstruct corrected ROC-AUC.

**New experiments are scientifically necessary before reporting metrics for the corrected scripts.** They are not necessary merely to retain the explicitly identified historical baseline. No corrected-code accuracy, timing, throughput, ROC-AUC or emissions are claimed here.

The same script and fixed dataset ordering are required for repeatability. In particular, the VWW scripts sample unsorted directory listings, so a fixed random seed alone does not establish identical image selection across machines.

## Combined CSV Precision

All 18 records in `mlperf_tiny_all_devices_summary.csv` were compared to the corresponding recorded platform summaries using `Device`, `Workload` and `Model` as the unique key. All non-emissions fields agree numerically. No platform, workload, sample count, accuracy, throughput, inference duration or other measurement was changed.

Before correction, all combined emissions values matched the source values rounded to three significant figures with decimal half-up rounding. This reduced numerical precision in 15 rows; the other three differed only in notation. The uppercase exponent and reduced precision are consistent with a formatting/re-export step, but the program that produced that file is not established by the available history.

The current `compile_all_devices.py` uses `.3e`, which retains four significant figures. Repository preparation copied each recorded source emissions string directly into the corresponding combined record. It did not calculate emissions, rerun a compiler that averages experimental runs, or alter any source CSV. Non-emissions cells and row order were preserved exactly as they stood before this correction.

IC means Image Classification; KWS means Keyword Spotting; VWW means Visual Wake Words. Values are in kgCO2e.

| Device | Workload/configuration | Combined before | Recorded source and corrected combined | Change |
|---|---|---|---|---|
| Ryzen 7 6800H | IC (full precision) | `1.62E-05` | `1.624e-05` | Source precision restored |
| Ryzen 7 6800H | KWS (full precision) | `2.67E-06` | `2.673e-06` | Source precision restored |
| Ryzen 7 6800H | VWW (full precision) | `8.70E-05` | `8.702e-05` | Source precision restored |
| ESP32-S3 | IC (INT8) | `1.01E-06` | `1.014e-06` | Source precision restored |
| ESP32-S3 | KWS (INT8) | `1.13E-06` | `1.130e-06` | Notation only |
| ESP32-S3 | VWW (INT8) | `4.41E-06` | `4.405e-06` | Source precision restored |
| RTX 4090 | IC (full precision) | `1.04E-04` | `1.036e-04` | Source precision restored |
| RTX 4090 | KWS (full precision) | `8.27E-06` | `8.266e-06` | Source precision restored |
| RTX 4090 | VWW (full precision) | `4.92E-04` | `4.918e-04` | Source precision restored |
| Raspberry Pi 5 | IC (full precision) | `8.19E-06` | `8.190e-06` | Notation only |
| Raspberry Pi 5 | IC (INT8) | `1.22E-07` | `1.222e-07` | Source precision restored |
| Raspberry Pi 5 | KWS (full precision) | `6.57E-07` | `6.570e-07` | Notation only |
| Raspberry Pi 5 | KWS (INT8) | `1.81E-07` | `1.812e-07` | Source precision restored |
| Raspberry Pi 5 | VWW (full precision) | `3.94E-05` | `3.943e-05` | Source precision restored |
| Raspberry Pi 5 | VWW (INT8) | `6.09E-07` | `6.087e-07` | Source precision restored |
| Galaxy S26 Ultra | IC (INT8) | `4.35E-10` | `4.346e-10` | Source precision restored |
| Galaxy S26 Ultra | KWS (INT8) | `1.40E-09` | `1.395e-09` | Source precision restored |
| Galaxy S26 Ultra | VWW (INT8) | `3.17E-09` | `3.174e-09` | Source precision restored |

The source files are:

- [CPU summary](MLPerf_Tiny_PC_CPU/mlperf_tiny_PC_CPU_summary.csv)
- [GPU summary](MLPerf_Tiny_PC_GPU/mlperf_tiny_PC_GPU_summary.csv)
- [Raspberry Pi 5 summary](MLPerf_Tiny_RaspberryPi5/mlperf_tiny_RaspberryPi5_summary.csv)
- [Android summary](MLPerf_Tiny_Android/mlperf_tiny_Android_summary.csv)
- [ESP32-S3 summary](MLPerf_Tiny_ESP32-S3/mlperf_tiny_ESP32S3_summary.csv)

## Timing and Energy Scope

CPU, GPU, Raspberry Pi and Android TinyML results time the complete Python prediction loop per workload repetition, including runtime and output handling. ESP32-S3 results sum device-side `Invoke()` durations and exclude host work, serial transfer and the firmware's 500 ms command delay. These timing scopes differ.

CodeCarbon estimates use platform-specific power estimates and regional carbon intensities. ESP32-S3 estimates apply the documented assumed chip power to invocation time. Neither method establishes direct whole-device electricity measurements.

## Retained Local Data

The Android and Raspberry Pi exported KWS `benchmark_data/*.npy` arrays and the CPU VWW `results_new/` files were removed from the current Git index only. Their local files were retained. Earlier commits still contain those files; Git history was not rewritten.

## CodeCarbon Geographical Metadata Cleanup

The current public copies of the 19 historical CodeCarbon `emissions.csv` files (15 TinyML and four LLM files) have been privacy-reduced by removing only `latitude`, `longitude`, `country_name`, `country_iso_code`, `region` and `cloud_region`. They use the approved privacy-reduced copies from the standalone EdgeAI repositories.

All 937 historical measurement rows and their ordering are preserved. Every retained CSV cell, including scientific measurements, numerical precision, timestamps, run identifiers and experiment identifiers, is unchanged. No benchmarks were rerun and no emissions or energy values were recalculated.

This cleanup applies to the current CSV copies. Historical Git commits have not been cleaned or rewritten.
