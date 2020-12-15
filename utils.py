from tensorflow.keras.metrics import Metric, FalsePositives, FalseNegatives
import numpy as np
import tensorflow as tf
class MetricWrapper(Metric):
  def __init__(self, keras_metric):
    super(MetricWrapper, self).__init__(name="wrapped_" + keras_metric.name)
    self.metric = keras_metric
  def update_state(self, y_true, y_pred, sample_weight=None):

    y_pred = y_pred[:, :, 1]
    self.metric.update_state(y_true, y_pred, sample_weight)
  def result(self):
    return self.metric.result()
  def reset_states(self):
    self.metric.reset_states()
  def get_config(self):
    self.metric.get_config()

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

wrapped_metrics = list(map(lambda m: MetricWrapper(m), METRICS))
