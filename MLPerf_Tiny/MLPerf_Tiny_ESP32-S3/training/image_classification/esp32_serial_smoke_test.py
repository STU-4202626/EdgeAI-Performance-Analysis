import time
import struct

import numpy as np
import serial


PORT = "COM8"
BAUD = 921600

INPUT_SIZE = 32 * 32 * 3
OUTPUT_SIZE = 10


def read_exact(ser, size):
    data = bytearray()

    while len(data) < size:
        chunk = ser.read(size - len(data))

        if not chunk:
            raise TimeoutError(
                f"Timed out after receiving {len(data)} of {size} bytes"
            )

        data.extend(chunk)

    return bytes(data)


with serial.Serial(
    PORT,
    BAUD,
    timeout=5
) as ser:

    # Opening COM8 can reset some ESP32 boards.
    # Give the board time to boot and initialize TFLite Micro.
    print("Waiting for ESP32-S3 to boot...")
    time.sleep(3)

    # Remove boot messages / MicroPrintf text already waiting
    # in the PC's serial receive buffer.
    ser.reset_input_buffer()

    # -----------------------------------------------------
    # Test 1: PING
    # -----------------------------------------------------

    print("Sending PING...")

    ser.write(b"PING")
    ser.flush()

    response = read_exact(ser, 4)

    print("PING response:", response)

    if response != b"PONG":
        raise RuntimeError(
            f"Expected PONG but received {response!r}"
        )

    print("PING test passed.")

    # -----------------------------------------------------
    # Test 2: one inference
    # -----------------------------------------------------

    # Same artificial input we used previously:
    # 3072 INT8 zeros.
    test_input = np.zeros(
        INPUT_SIZE,
        dtype=np.int8
    )

    print("\nSending one 3072-byte INT8 tensor...")

    ser.write(b"INFR")
    ser.write(test_input.tobytes())
    ser.flush()

    # Response:
    # 4 bytes  -> RSLT
    # 8 bytes  -> signed 64-bit inference time, microseconds
    # 10 bytes -> INT8 output scores
    packet = read_exact(
        ser,
        4 + 8 + OUTPUT_SIZE
    )

    header = packet[0:4]

    if header != b"RSLT":
        raise RuntimeError(
            f"Expected RSLT but received {header!r}"
        )

    inference_us = struct.unpack(
        "<q",
        packet[4:12]
    )[0]

    scores = np.frombuffer(
        packet[12:22],
        dtype=np.int8
    ).copy()

    predicted_class = int(
        np.argmax(scores)
    )

    print("Inference response received.")
    print("Inference time:", inference_us, "us")
    print(
        "Inference time:",
        inference_us / 1000,
        "ms"
    )
    print("Raw INT8 outputs:", scores.tolist())
    print("Predicted class:", predicted_class)