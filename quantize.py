"""Quantization script."""

import os

import tensorflow as tf
import yaml
from absl import flags

from get_dataset import get_dataset
from model import get_tcn, get_cnn

# Define Flags.
flags.DEFINE_string(
    'exp_name',
    'HeySnipsSequence_batch_size64_epochs250_lr0.001_tcn_feat20_len3_stacks3_filters8_dil5_bn_skip',
    'Name of the experiment to be evaluated.')
flags.DEFINE_integer('exp_nr', 0, 'Experiment number.', lower_bound=0)

flags.DEFINE_string(
    'exp_root', 'exp',
    'Root directory of experiments.')
flags.DEFINE_string('data_root',
                    '.',
                    'Root directory of data.')

FLAGS = flags.FLAGS


def main(_):

  # get experiment folder and create dir for plots
  exp_folder = os.path.join(FLAGS.exp_root, FLAGS.exp_name,
                            'exp{}'.format(FLAGS.exp_nr))
  test_folder = os.path.join(exp_folder, 'test')
  tf.io.gfile.mkdir(test_folder)

  # get experiment FLAGS
  TRAINING_FLAGS = yaml.safe_load(
      tf.io.gfile.GFile(os.path.join(exp_folder, 'FLAGS.yml'), 'r')
  )

  # get dataset
  test_set, test_labels = get_dataset('.', TRAINING_FLAGS['num_feat'], TRAINING_FLAGS['slice_length'], type='test',
                                        return_sequences=TRAINING_FLAGS['return_sequences'])


  sequence_length = test_set.shape[1]
  feature_dim = test_set.shape[2]
  if TRAINING_FLAGS['model'] == 'tcn':

    model = get_tcn(sequence_length, feature_dim,
                    nb_filters=TRAINING_FLAGS['num_filters'],
                    nb_stacks=TRAINING_FLAGS['num_stacks'],
                    use_skip_connections=TRAINING_FLAGS['use_skip_connections'],
                    use_batch_norm=TRAINING_FLAGS['bn'],
                    return_sequences=False,#TRAINING_FLAGS['return_sequences'],
                    dilation_stages=TRAINING_FLAGS['dilation_stages'])
  elif TRAINING_FLAGS['model'] == 'cnn':
    model = get_cnn((sequence_length, feature_dim))


  else:
    assert False, 'Unknown model!'


  model(tf.zeros((1, sequence_length, feature_dim)))
  model.load_weights(os.path.join(exp_folder, 'model.h5'))
  model.compile()
  model.summary()

  #if tcn, we have to cut off the model above the strided slice since it is not supported in NNTool, we perform the last Dense layer as a matrix product
  if TRAINING_FLAGS['model'] == 'tcn':
    model = tf.keras.Model(inputs=[model.input], outputs=[model.get_layer(name='reshape_1').output])

  model.summary()
  converter = tf.lite.TFLiteConverter.from_keras_model(model)

  # Convert the model to the TensorFlow Lite format with quantization
  tflite_model_name = 'quant_model'
  quantize = True
  if (quantize):
    def representative_dataset():
      for i in range(100):
        yield [test_set[i].reshape(1, sequence_length, feature_dim)]

    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce full-int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representative_dataset
  tflite_model = converter.convert()
  model_path = os.path.join(exp_folder, tflite_model_name + '.tflite')
  open(model_path, 'wb').write(tflite_model)