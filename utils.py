from tensorflow.keras.metrics import Metric


class MetricWrapper(Metric):
  def __init__(self, keras_metric, dims=3):
    super(MetricWrapper, self).__init__(name=keras_metric.name)
    self.metric = keras_metric
    self.dims = dims
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.dims == 3:
      y_pred = y_pred[:, :, 1]
    else:
      y_pred = y_pred[:, 1]
    self.metric.update_state(y_true, y_pred, sample_weight)
  def result(self):
    return self.metric.result()
  def reset_states(self):
    self.metric.reset_states()
  def get_config(self):
    return {'dims': self.dims}

