import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import sys
sys.path.append('keras-tcn')
from tcn import TCN, tcn_full_summary


def get_tcn(seq_len, feature_dim):

  i = Input(shape=(seq_len,feature_dim))
  x = TCN(nb_filters=16,
          kernel_size=2,
          dilations=[1, 2, 4, 8],
          nb_stacks=3,
          use_skip_connections=True,
          return_sequences=True)(i)
  x = Dense(1, activation='sigmoid')(x)
  #x = tf.reshape(x, shape=[-1, seq_len])
  model = Model(inputs=[i], outputs=[x])
  return model