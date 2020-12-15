import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Layer, Softmax
from tensorflow.keras import Input, Model
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