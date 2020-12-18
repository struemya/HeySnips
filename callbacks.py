# Lint as: python3
"""Custom Keras Callbacks."""
import tensorflow as tf

class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
  """Custom ModelCheckpoint to save .h5 to cns"""

  def __init__(self,
               realfilepath,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.realfilepath = realfilepath

  def on_epoch_end(self,
                   epoch,
                   logs: None = None):
    super().on_epoch_end(epoch, logs=logs)
    tf.io.gfile.copy(self.filepath, self.realfilepath, overwrite=True)


