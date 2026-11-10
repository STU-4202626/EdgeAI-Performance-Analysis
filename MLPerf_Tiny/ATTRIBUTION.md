# Third-Party Source Attribution

This directory includes reference workloads, models and adapted source from [MLCommons MLPerf Tiny](https://github.com/mlcommons/tiny). The complete [Apache License 2.0 text](LICENSE.md) was copied unchanged from the local original MLPerf Tiny checkout at `Original_MlPerf_tiny/tiny/LICENSE.md`. Its text matches that checkout's upstream Git revision `1afd2c9820f795965a6134facd0b4dfae41ef23f`; the ignored checkout itself is not distributed here.

Original copyright headers and MLCommons, SiliconLabs and EEMBC references are retained. Source files adapted for this project carry prominent modification comments. Unchanged upstream copies retain their original notices; the presence of a copy under a different platform directory does not imply that its implementation was modified.

## Project Adaptations

EdgeAI adapts host benchmarking, classification evaluation, runtime selection, result collection and output paths. Directory-default and MobileNet-definition adaptations are marked in the relevant helpers. Original project helpers and compilation utilities are identified by their source rather than being blanket-relicensed through this attribution document.

The source-notice inventory below covers the confirmed missing notices identified during repository preparation, plus formatting-only changes needed for a clean whitespace check. Existing KWS and VWW benchmark headers already identify their adaptations. The four provenance-uncertain Android/Pi KWS/VWW working copies were not modified or staged; see [recorded-results provenance](RESULTS_PROVENANCE.md).

Some already tracked reference/backup source files remain in the historical repository. The existing Android Image Classification `og.py` receives only a modification notice; its executable content is unchanged. Other tracked backups that match upstream remain untouched.

## ESP32-S3 Firmware and Models

The firmware foundation includes the TensorFlow Authors' Apache-licensed source distributed with Espressif's [TensorFlow Lite Micro Hello World example](https://github.com/espressif/esp-tflite-micro/tree/master/examples/hello_world). Existing TensorFlow copyright headers are retained. The original TensorFlow 2020 Apache header was restored to the three adapted `main_functions.cc` files, whose Hello World interpreter/setup foundation was retained while adding workload-specific operators, UART transport, external PSRAM allocation and invocation timing. The main-component CMake files identify the added timer and UART dependencies.

Each embedded `main/model.cc` is a generated representation of its corresponding deployed MLPerf Tiny `.tflite` artifact. Embedded model bytes were not altered during documentation or attribution preparation.

The firmware manifests and lock files reference Espressif components. Downloaded `managed_components/` are excluded; their licences and third-party notices accompany the restored dependencies. This document does not replace those terms.

## Datasets and Academic Attribution

CIFAR-10, Speech Commands and Visual Wake Words remain subject to their dataset-specific terms and citation requirements. Dataset download sources are linked in the [project README](../README.md). Dataset files are excluded from the prepared current tree; earlier Git history is retained.

MLCommons requests citation of the MLPerf Tiny Benchmark paper when its materials are used in academic work. The upstream [MLPerf Tiny README](https://github.com/mlcommons/tiny) provides the citation. Project results are adapted research workloads, not official MLPerf Tiny submissions.

## Modification-Notice Inventory

All source changes made during this preparation are comments, retained/restored licence headers or whitespace. Executable benchmark logic is preserved. Formatting cleanup in `train_test_distill.py` also removes trailing whitespace from a standalone dataset-description string.

- `MLPerf_Tiny_Android/training/image_classification/eval_functions_eembc.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_Android/training/image_classification/mlperf_image_classification_Android_bench.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_Android/training/image_classification/og.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_Android/training/image_classification/test.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_Android/training/keyword_spotting/kws_util.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_Android/training/visual_wake_words/vww_model_updated.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/README.md`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/eval_functions_eembc.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/firmware/main/CMakeLists.txt`: Component-dependency adaptation notice; CMake commands retained.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/firmware/main/main_functions.cc`: Restored TensorFlow 2020 licence header and added adaptation notice; code retained.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/get_perf_samples_idxs.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/keras_model.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/mlperf_image_classification_ESP32S3_bench.py`: Project adaptation notice; executable content retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/test.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_ESP32-S3/training/image_classification/train_test_distill.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/README.md`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/build_ref.sh`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/eval_quantized_model.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/evaluate.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/firmware/main/CMakeLists.txt`: Component-dependency adaptation notice; CMake commands retained.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/firmware/main/main_functions.cc`: Restored TensorFlow 2020 licence header and added adaptation notice; code retained.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/get_dataset.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/keras_model.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/kws_util.py`: Project adaptation notice; executable content retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/make_bin_files.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/mk_cal_set.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/quantize.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/keyword_spotting/train.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/visual_wake_words/firmware/main/CMakeLists.txt`: Component-dependency adaptation notice; CMake commands retained.
- `MLPerf_Tiny_ESP32-S3/training/visual_wake_words/firmware/main/main_functions.cc`: Restored TensorFlow 2020 licence header and added adaptation notice; code retained.
- `MLPerf_Tiny_ESP32-S3/training/visual_wake_words/vww_model.py`: Formatting-only modification notice; original implementation retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_ESP32-S3/training/visual_wake_words/vww_model_updated.py`: Project adaptation notice; executable content retained. Trailing whitespace cleaned.
- `MLPerf_Tiny_PC_CPU/training/image_classification/eval_functions_eembc.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_CPU/training/image_classification/mlperf_image_classification_CPU_bench.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_CPU/training/image_classification/test.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_CPU/training/keyword_spotting/kws_util.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_CPU/training/visual_wake_words/vww_model_updated.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_GPU/training/image_classification/eval_functions_eembc.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_GPU/training/image_classification/mlperf_image_classification_RunPod_GPU.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_GPU/training/image_classification/test.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_GPU/training/keyword_spotting/kws_util.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_PC_GPU/training/visual_wake_words/vww_model_updated.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_RaspberryPi5/training/image_classification/eval_functions_eembc.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_RaspberryPi5/training/image_classification/mlperf_image_classification_Pi5_full.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_RaspberryPi5/training/image_classification/mlperf_image_classification_Pi5_quant.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_RaspberryPi5/training/image_classification/test.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_RaspberryPi5/training/keyword_spotting/kws_util.py`: Project adaptation notice; executable content retained.
- `MLPerf_Tiny_RaspberryPi5/training/visual_wake_words/vww_model_updated.py`: Project adaptation notice; executable content retained.
