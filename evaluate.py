# Lint as: python3
"""Evaluation script."""

import os
import pickle
import pprint

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
    'HeySnipsSequence_batch_size64_epochs250_lr0.0001_cnn_feat13_len3',
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
  test_set, test_labels = get_dataset(TRAINING_FLAGS['data_root'], TRAINING_FLAGS['num_feat'], TRAINING_FLAGS['slice_length'], type='test',
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
  parameters = model.count_params()
  pred_labels = model.predict(test_set, batch_size=64, use_multiprocessing=True, workers=8)

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

  res = {}
  for m in wrapped_metrics:
    m.update_state(y_true=test_labels, y_pred=pred_labels)
    res[m.name] = m.result().numpy()
  res['parameters'] = parameters

  with open(os.path.join(exp_folder, 'test/metrics.p'), 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
  pprint.pprint(res)
if __name__ == '__main__':
  app.run(main)
