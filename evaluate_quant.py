# Lint as: python3
"""Evaluate Quantization script"""

import os
import pickle
import pprint

import numpy as np
import tensorflow as tf
import yaml
from absl import app
from absl import flags

from get_dataset import get_dataset
from model import get_tcn, get_cnn
from utils import MetricWrapper

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
                    return_sequences=TRAINING_FLAGS['return_sequences'],
                    dilation_stages=TRAINING_FLAGS['dilation_stages'])
  elif TRAINING_FLAGS['model'] == 'cnn':
    model = get_cnn((sequence_length, feature_dim))


  else:
    assert False, 'Unknown model!'


  model(tf.zeros((1, sequence_length, feature_dim)))
  model.load_weights(os.path.join(exp_folder, 'model.h5'))
  model.compile()
  model.summary()
  # print(model.count_params())
  # weights = model.get_layer('dense').get_weights()
  # kernel = weights[0]
  # bias = weights[1]
  # scale = 128 / max(kernel.min(), kernel.max(), bias.min(), bias.max())
  # kernel_scaled = (kernel * scale).astype('int8')
  # bias_scaled = (bias * scale).astype('int8')


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
  model_path = os.path.join('/tmp', tflite_model_name + '.tflite')
  open(model_path, 'wb').write(tflite_model)




  tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
  tflite_interpreter.allocate_tensors()
  input_details = tflite_interpreter.get_input_details()
  output_details = tflite_interpreter.get_output_details()

  predictions = []
  for i in range(len(test_set)):
    val_batch = test_set[i]
    val_batch = np.expand_dims(val_batch, axis=0).astype(input_details[0]["dtype"])
    tflite_interpreter.set_tensor(input_details[0]['index'], val_batch)
    tflite_interpreter.allocate_tensors()
    tflite_interpreter.invoke()
    output = tflite_interpreter.get_tensor(output_details[0]['index'])
    predictions += [output]

  METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
  ]
  wrapped_metrics = list(map(lambda m: MetricWrapper(m, dims=2), METRICS))
  predictions = np.stack(predictions).squeeze()
  res = {}
  for m in wrapped_metrics:
    m.update_state(y_true=test_labels, y_pred=predictions)
    res[m.name] = m.result().numpy()


  with open(os.path.join(exp_folder, 'test/quant_metrics.p'), 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
  pprint.pprint(res)

if __name__ == '__main__':
  app.run(main)
