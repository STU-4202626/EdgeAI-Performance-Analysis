/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Modified for EdgeAI (2026): replaced the sine demonstration with
// workload-specific UART inference, PSRAM allocation and device timing.

#include <cstdint>
#include <cstddef>
#include <cstring>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"

namespace {

// ---------------------------------------------------------
// TFLite Micro objects
// ---------------------------------------------------------

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// ---------------------------------------------------------
// Tensor arena
// ---------------------------------------------------------

constexpr size_t kTensorArenaSize = 128 * 1024;

uint8_t* tensor_arena = nullptr;


// ---------------------------------------------------------
// Image Classification tensor sizes
// ---------------------------------------------------------

constexpr size_t kInputBytes = 32 * 32 * 3;  // 3072 INT8 values
constexpr size_t kOutputBytes = 10;          // 10 CIFAR-10 classes


// ---------------------------------------------------------
// Serial protocol
// ---------------------------------------------------------

// High-speed serial transport for benchmark tensors.
// Serial transfer time is excluded from device inference timing.
constexpr uart_port_t kUartPort = UART_NUM_0;
constexpr int kBaudRate = 921600;

// PC -> ESP32 commands
constexpr char kPingCommand[4] = {'P', 'I', 'N', 'G'};
constexpr char kInferCommand[4] = {'I', 'N', 'F', 'R'};

// ESP32 -> PC responses
constexpr char kPongResponse[4] = {'P', 'O', 'N', 'G'};
constexpr char kResultResponse[4] = {'R', 'S', 'L', 'T'};
constexpr char kErrorResponse[4] = {'E', 'R', 'R', '!'};


// ---------------------------------------------------------
// Read exactly the requested number of bytes.
//
// uart_read_bytes() is allowed to return fewer bytes than
// requested, so this function keeps reading until the full
// packet has arrived.
// ---------------------------------------------------------

bool ReadExact(uint8_t* destination, size_t length) {

  size_t received = 0;

  while (received < length) {

    const int bytes_read = uart_read_bytes(
        kUartPort,
        destination + received,
        length - received,
        portMAX_DELAY);

    if (bytes_read < 0) {
      return false;
    }

    received += static_cast<size_t>(bytes_read);
  }

  return true;
}


// ---------------------------------------------------------
// Send a raw block of bytes to the PC.
// ---------------------------------------------------------

void SendBytes(const void* data, size_t length) {

  uart_write_bytes(
      kUartPort,
      static_cast<const char*>(data),
      length);
}

}  // namespace


void setup() {

  MicroPrintf(
      "=== ESP32-S3 MLPerf Tiny Image Classification ===");

  MicroPrintf(
      "Model size: %d bytes",
      g_model_len);


  // -------------------------------------------------------
  // Load model
  // -------------------------------------------------------

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    MicroPrintf(
        "Model schema version %d does not match supported version %d",
        model->version(),
        TFLITE_SCHEMA_VERSION);

    return;
  }


  // -------------------------------------------------------
  // Register exactly the operators used by ResNet8
  // -------------------------------------------------------

  static tflite::MicroMutableOpResolver<6> resolver;

  if (resolver.AddAdd() != kTfLiteOk) {
    MicroPrintf("Failed to register ADD");
    return;
  }

  if (resolver.AddAveragePool2D() != kTfLiteOk) {
    MicroPrintf("Failed to register AVERAGE_POOL_2D");
    return;
  }

  if (resolver.AddConv2D() != kTfLiteOk) {
    MicroPrintf("Failed to register CONV_2D");
    return;
  }

  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("Failed to register FULLY_CONNECTED");
    return;
  }

  if (resolver.AddReshape() != kTfLiteOk) {
    MicroPrintf("Failed to register RESHAPE");
    return;
  }

  if (resolver.AddSoftmax() != kTfLiteOk) {
    MicroPrintf("Failed to register SOFTMAX");
    return;
  }


  // -------------------------------------------------------
  // Allocate tensor arena in PSRAM.
  //
  // Our controlled test showed this configuration averaged
  // ~88.85 ms versus ~105.93 ms using internal SRAM.
  // -------------------------------------------------------

  tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(
          kTensorArenaSize,
          MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

  if (tensor_arena == nullptr) {

    MicroPrintf(
        "Failed to allocate tensor arena");

    return;
  }


  // -------------------------------------------------------
  // Create TFLite Micro interpreter
  // -------------------------------------------------------

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {

    MicroPrintf(
        "AllocateTensors() failed");

    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf(
      "Tensor arena used: %d bytes",
      static_cast<int>(interpreter->arena_used_bytes()));

  MicroPrintf(
      "Input: [1,32,32,3] INT8");

  MicroPrintf(
      "Output: [1,10] INT8");


  // -------------------------------------------------------
  // Configure UART0.
  //
  // This is the same UART currently appearing as COM8.
  // The benchmark communicates at 921600 baud.
  // -------------------------------------------------------

  uart_config_t uart_config = {};

  uart_config.baud_rate = kBaudRate;
  uart_config.data_bits = UART_DATA_8_BITS;
  uart_config.parity = UART_PARITY_DISABLE;
  uart_config.stop_bits = UART_STOP_BITS_1;
  uart_config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
  uart_config.source_clk = UART_SCLK_DEFAULT;

  if (uart_param_config(
          kUartPort,
          &uart_config) != ESP_OK) {

    MicroPrintf("UART configuration failed");
    return;
  }

  // Your ESP32-S3 console already reported:
  // GPIO 43 = TX
  // GPIO 44 = RX

  if (uart_set_pin(
          kUartPort,
          43,
          44,
          UART_PIN_NO_CHANGE,
          UART_PIN_NO_CHANGE) != ESP_OK) {

    MicroPrintf("UART pin configuration failed");
    return;
  }

  // Install an interrupt-driven UART receive buffer.
  // 8192 bytes comfortably holds our 3072-byte image tensor.
  if (!uart_is_driver_installed(kUartPort)) {

    if (uart_driver_install(
            kUartPort,
            8192,
            0,
            0,
            nullptr,
            0) != ESP_OK) {

      MicroPrintf("UART driver installation failed");
      return;
    }
  }

  // Discard any old bytes left over from boot/terminal use.
  uart_flush_input(kUartPort);

  MicroPrintf(
      "Serial inference interface ready");
}


void loop() {

  // -------------------------------------------------------
  // Wait for a four-byte command from the PC.
  // -------------------------------------------------------

  uint8_t command[4];

  if (!ReadExact(command, sizeof(command))) {
    return;
  }


  // -------------------------------------------------------
  // PING
  //
  // Used by the Python program to verify that COM8 is
  // connected to the correct firmware before benchmarking.
  // -------------------------------------------------------

  if (std::memcmp(
          command,
          kPingCommand,
          sizeof(command)) == 0) {

    SendBytes(
        kPongResponse,
        sizeof(kPongResponse));

    return;
  }


  // -------------------------------------------------------
  // INFR
  //
  // Immediately after INFR, the PC sends exactly 3072
  // INT8 bytes representing one CIFAR-10 image.
  // -------------------------------------------------------

  if (std::memcmp(
          command,
          kInferCommand,
          sizeof(command)) == 0) {

    // Receive directly into the TFLite input tensor.
    //
    // Serial transfer is therefore completed BEFORE the
    // inference timer starts.
    if (!ReadExact(
            reinterpret_cast<uint8_t*>(input->data.int8),
            kInputBytes)) {

      SendBytes(
          kErrorResponse,
          sizeof(kErrorResponse));

      return;
    }


    // -----------------------------------------------------
    // Measure only model inference.
    // -----------------------------------------------------

    const int64_t start_us =
        esp_timer_get_time();

    const TfLiteStatus invoke_status =
        interpreter->Invoke();

    const int64_t end_us =
        esp_timer_get_time();


    if (invoke_status != kTfLiteOk) {

      SendBytes(
          kErrorResponse,
          sizeof(kErrorResponse));

      return;
    }

    const int64_t inference_us =
        end_us - start_us;


    // -----------------------------------------------------
    // Response packet
    //
    // 4 bytes  : "RSLT"
    // 8 bytes  : inference time in microseconds
    // 10 bytes : raw INT8 Softmax output
    //
    // Total = 22 bytes
    // -----------------------------------------------------

    SendBytes(
        kResultResponse,
        sizeof(kResultResponse));

    SendBytes(
        &inference_us,
        sizeof(inference_us));

    SendBytes(
        output->data.int8,
        kOutputBytes);

    return;
  }


  // Unknown command.
  SendBytes(
      kErrorResponse,
      sizeof(kErrorResponse));
}