# ESP32-S3 Keyword Spotting Firmware

This firmware runs the INT8 DS-CNN model for the project's adapted MLPerf Tiny Keyword Spotting workload. The Python host prepares input tensors, sends them over serial, computes classification metrics and writes results. Model inference runs on the ESP32-S3.

These are adapted research workloads, not official MLPerf Tiny submissions.

## Configuration

| Component | Configuration |
|---|---|
| Board used for the project | Waveshare ESP32-S3 with 16 MB flash and 8 MB Octal PSRAM |
| CPU | Dual-core Xtensa LX7, configured at 240 MHz |
| Framework | ESP-IDF 6.0.3 |
| Inference components | `esp-tflite-micro` 1.4.0 (TensorFlow Lite Micro) and `esp-nn` 1.3.2 |
| PSRAM | Octal mode at 80 MHz |
| Model artifact | `../trained_models/kws_ref_model.tflite` |
| Embedded model | `main/model.cc`, 53,936 model bytes |
| Input | INT8, shape `[1, 49, 10, 1]`, 490 bytes |
| Output | INT8, 12 quantised Softmax values |
| Tensor arena | 128 KiB, allocated in external PSRAM |
| Serial transport | UART0, 921600 baud, 8N1, GPIO43 TX and GPIO44 RX |

[`sdkconfig.defaults`](sdkconfig.defaults) preserves the ESP32-S3 target, QIO 16 MB flash, Octal PSRAM at 80 MHz, 240 MHz CPU and custom partition settings. It also disables the interrupt watchdog and task-watchdog initialization used by these benchmark projects.

The defaults place the partition table at `0xc000`. [`partitions.csv`](partitions.csv) provides two `0x1E0000` OTA application slots within the first 4 MiB of flash. The layout does not allocate the board's entire 16 MB capacity.

[`main/idf_component.yml`](main/idf_component.yml) declares the inference dependency. [`dependencies.lock`](dependencies.lock) records the resolved component and ESP-IDF versions. Retain these files alongside the source. Generated `sdkconfig`, `build/` and downloaded `managed_components/` are excluded from Git.

## Build and Flash

The following commands document the intended procedure. A fresh-clone or clean build has not been validated during the repository audit.

1. Activate the **ESP-IDF 6.0.3** environment.
2. Open this workload's `firmware/` directory.
3. Select the target and build:

   ```text
   idf.py set-target esp32s3
   idf.py build
   ```

   The component manager resolves dependencies using the manifest and lock file. Initial setup may require network access to retrieve the components.

4. Flash this workload's firmware, replacing `COM8` with the actual board port:

   ```text
   idf.py -p COM8 flash
   ```

5. Close any serial monitor before starting the host benchmark, which needs exclusive access to the port.

Flash the firmware matching the host workload. `PING`/`PONG` verifies the connection but does not identify which model is flashed.

## Host Benchmark

Run [`../mlperf_keyword_spotting_ESP32S3_bench.py`](../mlperf_keyword_spotting_ESP32S3_bench.py) from the parent `keyword_spotting/` directory. Model, dataset and result paths are relative to the working directory.

The host currently sets `SERIAL_PORT = "COM8"` and `SERIAL_BAUD = 921600`. Adjust the port for the local machine.

Common Python dependencies are `pyserial`, `numpy`, `pandas`, `matplotlib` and `scikit-learn`. Keyword spotting additionally requires TensorFlow and tensorflow-datasets. Its dataset helpers use TensorFlow/Keras and `tensorflow.lite.experimental.microfrontend.python.ops.audio_microfrontend_op`, so the environment must provide that module. These dependencies prepare features on the host; inference runs on the ESP32-S3. Python dependency versions are not pinned here, and a fresh host environment has not been validated during the audit.

Prepare Speech Commands v0.02 in a TensorFlow Datasets data directory. The host loads the first 1000 test samples, prepares 49 × 10 feature tensors and evaluates 12 output classes. Pass `--data_dir` to select the prepared cache; the default is `~/data`. The loader may attempt preparation or downloading if data is absent. `--bg_path` selects the background-noise location and defaults to the workload directory.

From the workload directory:

```text
python mlperf_keyword_spotting_ESP32S3_bench.py --data_dir PATH_TO_PREPARED_TFDS_DATA
```

The host writes `results/ESP32S3_keyword_spotting_benchmark_results.csv`, predictions and figures. It uses `../results_helper.py` to save the per-run CSV and `../esp32_energy_helper.py` to estimate energy and carbon emissions.

## Serial Protocol and Timing

The host sends `PING` and expects `PONG`. For each inference, it sends `INFR` followed by exactly 490 INT8 input bytes. The device returns:

| Field | Length |
|---|---:|
| `RSLT` response marker | 4 bytes |
| Signed little-endian inference duration in microseconds | 8 bytes |
| INT8 model output | 12 bytes |

`ERR!` indicates a firmware-side error.

The timer in `main/main_functions.cc` surrounds only `interpreter->Invoke()`. Host preprocessing, serial transfers, output handling and the 500 ms delay after each protocol command in `main/main.cc` are excluded. Reported throughput describes model invocation performance; end-to-end serial processing includes these additional costs.

The host performs one warm-up followed by three measured runs. Each run repeats complete 1000-sample workloads until at least 30 seconds of accumulated device invocation time has been reached. Reported inference time is normalized to one complete workload; throughput is the sample count divided by that time.

The energy helper applies an assumed power of **0.29007 W** and a carbon-intensity factor of **0.707686 kgCO2e/kWh** to invocation time. These estimates are not direct measurements of board or host electricity consumption.

## Attribution

The workload and reference model derive from [MLCommons MLPerf Tiny](https://github.com/mlcommons/tiny). Firmware development used TensorFlow Lite Micro and Espressif's Hello World example as a foundation. TensorFlow copyright headers are retained in the relevant source files, and adapted files carry modification notices.

See the [Apache 2.0 licence](../../../../LICENSE.md) and [source attribution](../../../../ATTRIBUTION.md). Restored dependencies remain subject to their respective licences.
