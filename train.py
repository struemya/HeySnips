# Lint as: python3
"""Training script."""
import os

import numpy as np
import tensorflow as tf
import yaml
from absl import app
from absl import flags
from sklearn.utils import class_weight
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.callbacks import ModelCheckpoint

from get_dataset import get_dataset
from model import get_tcn, get_cnn
from utils import MetricWrapper

# Define Flags.
flags.DEFINE_string('data_root',
                    '.',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    'exp',
                    'Root directory of experiments.')
flags.DEFINE_string('exp_folder',
                    '',
                    'When given overrides exp_root. Used with XManager.')

flags.DEFINE_enum('dataset', 'HeySnipsSequence',
                  ['HeySnipsSequence'],
                  'Dataset used during training.')
flags.DEFINE_integer('batch_size',
                     64,
                     'Batch size used during training.',
                     lower_bound=1)
flags.DEFINE_integer('epochs',
                     250,
                     'Maximum number of epochs.',
                     lower_bound=1)
flags.DEFINE_float('lr',
                   0.0001,
                   'Learning rate used during training.',
                   lower_bound=0.0)
flags.DEFINE_integer('patience',
                     150,
                     'Hyperparameter for early stopping.',
                     lower_bound=1)
flags.DEFINE_integer('nr_classes',
                     2,
                     'Number of classes.',
                     lower_bound=1)
flags.DEFINE_integer('num_feat',
                     20,
                     'Number of input audio features',
                     lower_bound=1)
flags.DEFINE_integer('dilation_stages',
                     5,
                     'Number of Blocks per Res unit, max dilation will ve 2**(dilation_stages - 1)')
flags.DEFINE_enum('model',
                  'tcn',
                  ['tcn', 'cnn'],
                  'Model used during training.')

flags.DEFINE_bool('bn',
                  True,
                  'Whether to use batch normalization in resnet.')

flags.DEFINE_integer('num_stacks',
                     3,
                     'Number of Stacks in TCN',
                     lower_bound=1)
flags.DEFINE_integer('num_filters',
                     16,
                     'Number of Filters in Convolutions',
                     lower_bound=1)

flags.DEFINE_integer('slice_length',
                     3,
                     'Length of audio slice per training example')
flags.DEFINE_bool('use_skip_connections',
                  True,
                  'Whether to use skip connections or not')
flags.DEFINE_bool('return_sequences', False,
                  'Sets if model should return single label or full sequence of labels for all time steps')

flags.DEFINE_bool('debug',
                  False,
                  'Debug mode. Eager execution.')

FLAGS = flags.FLAGS


def get_experiment_folder():
  """Create string with experiment name and get number of experiment."""

  # create exp folder
  exp_name = '_'.join([
    FLAGS.dataset, 'batch_size' + str(FLAGS.batch_size),
                   'epochs' + str(FLAGS.epochs), 'lr' + str(FLAGS.lr), FLAGS.model, 'feat' + str(FLAGS.num_feat), 'len' + str(FLAGS.slice_length)
  ])
  if FLAGS.model == 'tcn':
    exp_name = '_'.join([
      exp_name, 'stacks' + str(FLAGS.num_stacks),
                'filters' + str(FLAGS.num_filters),
                'dil' + str(FLAGS.dilation_stages)
    ])
    if FLAGS.bn:
      exp_name = '_'.join([
        exp_name, 'bn'
      ])
    if FLAGS.use_skip_connections:
      exp_name = '_'.join([
        exp_name, 'skip'
      ])
  # get number of experiment
  dirs = []
  exp_folder = os.path.join(FLAGS.exp_root, exp_name)
  tf.io.gfile.mkdir(exp_folder)
  for i in tf.io.gfile.listdir(exp_folder):
    if tf.io.gfile.isdir(os.path.join(exp_folder, i)):
      dirs += [i]
  exp_nr = len(dirs)
  exp_folder = os.path.join(exp_folder, 'exp{}'.format(exp_nr))
  tf.io.gfile.mkdir(exp_folder)

  return exp_folder, exp_name


def main(_):
  tf.config.experimental_run_functions_eagerly(FLAGS.debug)

  # get experiment folder
  if FLAGS.exp_folder == '':
    exp_folder, exp_name = get_experiment_folder()
  else:
    exp_folder = FLAGS.exp_folder

  # save FLAGS to yml
  tf.io.gfile.makedirs(exp_folder)
  yaml.dump(
    FLAGS.flag_values_dict(),
    tf.io.gfile.GFile(os.path.join(exp_folder, 'FLAGS.yml'), 'w'),
  )

  # get dataset
  train_set, train_labels = get_dataset(FLAGS.data_root, FLAGS.num_feat, FLAGS.slice_length,
                                        return_sequences=FLAGS.return_sequences)

  # optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)

  # loss function
  loss_func = sparse_categorical_crossentropy

  sequence_length = train_set.shape[1]
  feature_dim = train_set.shape[2]
  if FLAGS.model == 'tcn':

    model = get_tcn(sequence_length, feature_dim,
                    nb_filters=FLAGS.num_filters,
                    nb_stacks=FLAGS.num_stacks,
                    use_skip_connections=FLAGS.use_skip_connections,
                    use_batch_norm=FLAGS.bn,
                    return_sequences=FLAGS.return_sequences,
                    dilation_stages=FLAGS.dilation_stages)
  elif FLAGS.model == 'cnn':
    model = get_cnn((sequence_length, feature_dim))

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

  #wrap binary metric to make them work with sparse categorical crossentropy
  wrapped_metrics = list(map(lambda m: MetricWrapper(m), METRICS))
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=exp_folder, write_graph=True),
    tf.keras.callbacks.EarlyStopping(patience=FLAGS.patience),
    ModelCheckpoint(filepath=os.path.join(exp_folder, 'model.h5'), save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=100)
    ]

  if FLAGS.return_sequences:
    model.compile(
      optimizer=optimizer,
      loss=loss_func,
      metrics=['accuracy'], #add metrics if wished
      sample_weight_mode='temporal'
    )

    class_weights = class_weight.compute_class_weight('balanced',
                                                     classes=[0, 1],
                                                     y=train_labels.flatten())

    #temporal sample weighting: give higher weight to timesteps that are labelled as 1 (rare class)
    sample_weight = train_labels * class_weights[1] + class_weights[0] - train_labels * class_weights[0]
    train_labels = np.expand_dims(train_labels, axis=-1) #necessary because tf complains otherwise that data is not temporal

    # train model
    history = model.fit(
      train_set, train_labels,
      epochs=FLAGS.epochs,
      batch_size=FLAGS.batch_size,
      validation_split=0.25,
      callbacks=callbacks,
      sample_weight=sample_weight
    )
  else:
    model.compile(
      optimizer=optimizer,
      loss=loss_func,
      metrics=['accuracy']
    )

    #use class weights to counteract class imbalance
    class_weights = class_weight.compute_class_weight('balanced',
                                                     classes=[0, 1],
                                                     y=train_labels)
    class_weights = {i: class_weights[i] for i in range(2)}
    history = model.fit(
      train_set, train_labels,
      epochs=FLAGS.epochs,
      batch_size=FLAGS.batch_size,
      validation_split=0.25,
      callbacks=callbacks,
      class_weight=class_weights
    )

  # save history to yaml
  yaml.dump(
    history.history,
    tf.io.gfile.GFile(os.path.join(exp_folder, 'history.yml'), 'w'))


if __name__ == '__main__':
  app.run(main)
