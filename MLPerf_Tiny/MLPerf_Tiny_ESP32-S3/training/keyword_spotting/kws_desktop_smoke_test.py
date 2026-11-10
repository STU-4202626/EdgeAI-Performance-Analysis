import numpy as np
import tensorflow as tf


MODEL_PATH = "trained_models/kws_ref_model.tflite"

INPUT_SHAPE = (1, 49, 10, 1)
INPUT_ZERO_POINT = 83

OUTPUT_SCALE = 0.00390625
OUTPUT_ZERO_POINT = -128


# A real feature value of zero maps to INT8 value 83.
input_data = np.full(
    INPUT_SHAPE,
    INPUT_ZERO_POINT,
    dtype=np.int8
)


interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


interpreter.set_tensor(
    input_details["index"],
    input_data
)

interpreter.invoke()


raw_output = interpreter.get_tensor(
    output_details["index"]
)[0]


# Convert the quantized output back to floating point.
output = OUTPUT_SCALE * (
    raw_output.astype(np.float32)
    - OUTPUT_ZERO_POINT
)


print("Raw output:", raw_output)
print("Dequantized output:", output)
print("Predicted class:", np.argmax(raw_output))