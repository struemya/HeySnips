import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Softmax, Conv2D, BatchNormalization, GlobalAveragePooling2D, MaxPool2D, ReLU
from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l1
# import sys
# sys.path.append('keras-tcn')
from tcn import TCN, tcn_full_summary
from tcn import MyConv1D

def get_tcn(seq_len, feature_dim, output_dim=2, nb_filters=8, nb_stacks=3, use_skip_connections=True, return_sequences=True, use_batch_norm=True, dilation_stages=5):

  dilations = [2**x for x in range(dilation_stages)]
  i = Input(shape=(seq_len, feature_dim))
  x = Reshape(target_shape=(1, seq_len, feature_dim))(i)
  x = MyConv1D(filters=nb_filters,
               kernel_size=3,
               name='initial_conv',
               kernel_initializer='glorot_normal')(x)
  x = TCN(nb_filters=nb_filters,
          kernel_size=3,
          dilations=dilations,
          nb_stacks=nb_stacks,
          use_skip_connections=use_skip_connections,
          return_sequences=return_sequences,
          use_batch_norm=use_batch_norm,
          kernel_initializer='glorot_normal')(x)
  if return_sequences:
    x = Reshape(target_shape=(seq_len, nb_filters))(x)
  else:
    x = Reshape(target_shape=(nb_filters,))(x)
  x = Dense(output_dim)(x)
  x = Softmax()(x)
  model = Model(inputs=[i], outputs=[x])

  return model

def get_cnn(input_shape):
  model = tf.keras.models.Sequential()
  shape = (input_shape[0], input_shape[1], 1)
  model.add(Reshape(shape))
  model.add(Conv2D(filters=3, kernel_size=(3, 3), padding="same", input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(MaxPool2D((2, 2)))

  model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(MaxPool2D((2, 2)))

  model.add(Conv2D(filters=48, kernel_size=(3, 3), padding='same', strides=(2, 2)))
  model.add(BatchNormalization())
  model.add(ReLU())

  model.add(GlobalAveragePooling2D())

  model.add(Flatten())

  model.add(Dense(8, kernel_regularizer=(l1(0))))
  model.add(ReLU())

  model.add(Dense(2))
  model.add(Softmax())
  return model