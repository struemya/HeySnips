import json
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
from get_dataset import get_dataset
from model import get_cnn

train_set, train_labels = get_dataset('.',13,3,return_sequences=False)

model = get_cnn(train_set[0].shape)
model(tf.expand_dims(tf.zeros_like(train_set[0]), axis=0))
model.compile(loss='sparse_categorical_crossentropy')
model.fit(train_set, train_labels, batch_size=1, steps_per_epoch=10)
print(model.count_params())

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model to the TensorFlow Lite format with quantization
tflite_model_name = 'cnn_baseline'
quantize = True
if (quantize):
    def representative_dataset():
        for i in range(1):
          yield([train_set[i].reshape(1,299, 13)])
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce full-int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

open(tflite_model_name + '.tflite', 'wb').write(tflite_model)

"""#### This function here takes in the model and outputs an header file we will import into the TFLite example project folder. (/Core/Inc/)"""

# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data) :

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str

c_model_name = 'MFCC'
# Write TFLite model to a C source (or header) file
with open(c_model_name + '.h', 'w') as file:
    file.write(hex_to_c_array(tflite_model, c_model_name))

"""#### Let's have a look at the network we just generated"""

tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model_name + '.tflite')
tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

"""#### Next let's see how the performance of the network is"""

predictions = []
input_scale, input_zero_point = input_details[0]["quantization"]
for i in range(100):#len(test_set)):
    val_batch = test_set[i]
    #We must convert the data into int8 format before invoking inference.
  #  val_batch = val_batch / input_scale + input_zero_point
    val_batch = np.expand_dims(val_batch, axis=0).astype(input_details[0]["dtype"])
    tflite_interpreter.set_tensor(input_details[0]['index'], val_batch)
    tflite_interpreter.allocate_tensors()
    tflite_interpreter.invoke()

    tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
    #print("Prediction results shape:", tflite_model_predictions.shape)
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    predictions+= [output]

sum = 0
#m = tf.keras.metrics.BinaryAccuracy()
m = tf.keras.Accuracy()
m.update_state(test_labels[:100],np.squeeze(np.asarray(predictions)))

# for i in range(len(predictions)):
#     if (predictions[i] == test_labels[i]):
#         sum = sum + 1
# accuracy_score = sum / 100
print("Accuracy of quantized to int8 model is {}%".format(m.result().numpy()*100))
# print("Compared to float32 accuracy of {}%".format(score[1]*100))
# print("We have a change of {}%".format((accuracy_score-score[1])*100))