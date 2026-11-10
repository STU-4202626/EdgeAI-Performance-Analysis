import struct
import time

import numpy as np
import serial


SERIAL_PORT = "COM8"
SERIAL_BAUD = 921600
SERIAL_TIMEOUT = 5.0

INPUT_SHAPE = (1, 49, 10, 1)
OUTPUT_SIZE = 12

# A real feature value of zero maps to INT8 value 83 for this quantized KWS model.
input_data = np.full(
    INPUT_SHAPE,
    83,
    dtype=np.int8
)


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
    SERIAL_PORT,
    SERIAL_BAUD,
    timeout=SERIAL_TIMEOUT
) as ser:

    print("Waiting for ESP32-S3 to boot...")
    time.sleep(3)

    ser.reset_input_buffer()

    # Check that the expected firmware is responding.
    ser.write(b"PING")
    ser.flush()

    response = read_exact(ser, 4)

    print("PING response:", response)

    if response != b"PONG":
        raise RuntimeError("ESP32-S3 did not return PONG")

    # Send one KWS input tensor.
    ser.write(b"INFR")
    ser.write(input_data.tobytes())
    ser.flush()

    header = read_exact(ser, 4)

    if header != b"RSLT":
        raise RuntimeError(
            f"Expected RSLT but received {header!r}"
        )

    inference_us = struct.unpack(
        "<q",
        read_exact(ser, 8)
    )[0]

    output = np.frombuffer(
        read_exact(ser, OUTPUT_SIZE),
        dtype=np.int8
    ).copy()

    print("Inference time:", inference_us, "us")
    print("Inference time:", inference_us / 1000.0, "ms")
    print("Raw output:", output)
    print("Predicted class:", np.argmax(output))