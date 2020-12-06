import json
import numpy as np
from sklearn.utils import class_weight
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
#Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
# from tensorflow.python.compiler.mlcompute import mlcompute
#
# # Select CPU device.
# mlcompute.set_mlc_device(device_name='gpu')
import sys
sys.path.append('keras-tcn')



from get_dataset import load_data_time_series,load_data_fixed_length, compute_mfccs
from tcn import compiled_tcn
import pickle
from model import get_tcn
import matplotlib.pyplot as plt
from python_speech_features.base import logfbank

try:
  x_train_mfcc , y_train = pickle.load( open( "train_set.p", "rb" ))
  x_test_mfcc, y_test = pickle.load(open("test_set.p", "rb"))
  #assert 1
except:
  x_train, y_train, x_test, y_test = load_data_time_series()
  #x_train, y_train, x_test, y_test = load_data_fixed_length()
  x_train_mfcc = np.array(list(map(lambda x: logfbank(x), x_train)))
  pickle.dump((x_train_mfcc, y_train), open("train_set.p", "wb"))
  x_test_mfcc = np.array(list(map(lambda x: logfbank(x), x_test)))
  pickle.dump((x_test_mfcc, y_test), open("test_set.p", "wb"))

plt.figure()
for i in range(100):
  plt.plot(np.mean(x_train_mfcc[i], axis=-1), label='data', alpha=0.7)
  plt.plot(y_train[i], label='bool_audio')
  plt.legend()
  plt.show()


batchSize = 10
epochs = 50
trainsize = 10000

# train_set = tf.squeeze(x_train_mfcc/512 + 0.5)
train_labels =y_train
train_set = x_train_mfcc
#
# test_set = tf.squeeze(x_test_mfcc/512 + 0.5)
test_set = x_test_mfcc
test_labels = y_test


# METRICS = [
#       tf.keras.metrics.TruePositives(name='tp'),
#       tf.keras.metrics.FalsePositives(name='fp'),
#       tf.keras.metrics.TrueNegatives(name='tn'),
#       tf.keras.metrics.FalseNegatives(name='fn'),
#       tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#       tf.keras.metrics.Precision(name='precision'),
#       tf.keras.metrics.Recall(name='recall'),
#       tf.keras.metrics.AUC(name='auc'),
# ]
# opt = tf.keras.optimizers.Adam(0.0001)
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(16))
# model.add(tf.keras.layers.Dense(2, activation='softmax'))
# model.compile(opt, 'sparse_categorical_crossentropy', metrics=['accuracy'])
model = compiled_tcn(26, 2, 16, 3, [1, 2, 4, 8], 6, x_train_mfcc.shape[1], use_skip_connections=True, lr=.0001)
#model = get_tcn(train_set.shape[1], train_set.shape[2])
##model.compile(opt, 'binary_crossentropy', metrics=['accuracy'])#, sample_weight_mode="temporal")
class_weight = class_weight.compute_class_weight('balanced',
                                               classes=[0, 1],
                                                y=np.array(y_train).flatten())


print(np.unique(np.array(y_train).flatten()))
print(class_weight)

sample_weight = np.array(y_train, dtype='float32') * class_weight[1] + class_weight[0]

model.fit(train_set, np.array(train_labels), batchSize, epochs,  steps_per_epoch=trainsize/batchSize)#, sample_weight=sample_weight)

# # Reserve 10,000 samples for validation.
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-10000]
epochs = 5
batch_size = 10
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((train_set, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# # Prepare the validation dataset.
# val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_dataset = val_dataset.batch(batch_size)

# optimizer = tf.keras.optimizers.Adam(lr=0.001)
# def loss_fn(y_test, y_pred):
#   return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_test, y_pred))
#
#
# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))
#
#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#
#         # Open a GradientTape to record the operations run
#         # during the forward pass, which enables auto-differentiation.
#         with tf.GradientTape() as tape:
#
#             # Run the forward pass of the layer.
#             # The operations that the layer applies
#             # to its inputs are going to be recorded
#             # on the GradientTape.
#             logits = model(x_batch_train, training=True)  # Logits for this minibatch
#
#             # Compute the loss value for this minibatch.
#             loss_value = loss_fn(y_batch_train, logits)
#
#         # Use the gradient tape to automatically retrieve
#         # the gradients of the trainable variables with respect to the loss.
#         grads = tape.gradient(loss_value, model.trainable_weights)
#
#         # Run one step of gradient descent by updating
#         # the value of the variables to minimize the loss.
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#
#         # Log every 200 batches.
#         if step % 1 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )


model.summary()
score = model.evaluate(test_set, y_test)
pred = np.squeeze(model.predict(test_set))
#print(x_train_mfcc.shape)
#print(x_test_mfcc.shape)
