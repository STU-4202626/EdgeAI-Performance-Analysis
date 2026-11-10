import os
import tensorflow as tf


MODEL_PATH = "trained_models/vww_96_int8.tflite"


print("Model size:", os.path.getsize(MODEL_PATH), "bytes")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


print("\nInput tensor")
print("Name:", input_details["name"])
print("Shape:", input_details["shape"])
print("Shape signature:", input_details["shape_signature"])
print("Dtype:", input_details["dtype"])
print("Quantization:", input_details["quantization"])


print("\nOutput tensor")
print("Name:", output_details["name"])
print("Shape:", output_details["shape"])
print("Shape signature:", output_details["shape_signature"])
print("Dtype:", output_details["dtype"])
print("Quantization:", output_details["quantization"])


print("\nOperators")

ops = interpreter._get_ops_details()

for index, op in enumerate(ops):
    print(index, op["op_name"])


print("\nUnique operators")

unique_ops = []

for op in ops:
    if op["op_name"] not in unique_ops:
        unique_ops.append(op["op_name"])

for op in unique_ops:
    print(op)