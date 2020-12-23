import tensorflow as tf
from tensorflow import lite
import numpy as np

gap_scaling = 0.08045440167188644
tflite_scaling = 0.040069449692964554

test_value = 64


data = np.load('Circle_1s_wl32_doppl.npy')
print(np.shape(data))
test_sample = data[0,4:5,:,::2,:1]

# Generate sample_data.h file
test_file = "sample_data.h"
with open(test_file, "w+") as f:
	f.write("const int8_t test_cnn_data[7872] = {")
	for i in range(246):
		f.write("\r\n    ")
		for j in range(32):
			f.write("(int8_t)%i, " % test_value)#((test_sample[0, j, i, 0]/gap_scaling)))
	f.write("\r\n};\r\n\r\n\r\n")

interpreter = tf.lite.Interpreter(model_path="CNN/quant_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
# print(input_shape)
# print(test_sample.shape)
for i in range(0,10):
	input_data = (i*np.ones(input_shape)).astype(np.float32)#(test_sample/tflite_scaling-128).astype(np.int8)
	interpreter.set_tensor(input_details[0]['index'], input_data)

	interpreter.invoke()

	# The function `get_tensor()` returns a copy of the tensor data.
	# Use `tensor()` in order to get a pointer to the tensor.
	output_data = interpreter.get_tensor(output_details[0]['index'])

	print(str(i+128) + ": " + str((output_data+128)/2))
