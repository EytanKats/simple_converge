import tensorflow as tf
from tf_callbacks.CallbacksEnum import CallbacksEnum


class Callbacks(object):

    def __init__(self, settings):
        self.settings = settings
        self.callbacks_dict = {CallbacksEnum.CHECKPOINT: self._checkpoint,
                               CallbacksEnum.LOGGER: self._csv_logger,
                               CallbacksEnum.EARLY_STOPPING: self._early_stopping,
                               CallbacksEnum.REDUCE_LR_ON_PLATEAU: self._reduce_lr_onplateu}

    def get_callbacks(self):
        callbacks = list()
        for callback in self.settings.callbacks:
            callbacks.append(self.callbacks_dict[callback]())

        return callbacks

    def _checkpoint(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath=self.settings.checkpoint_weights_path,
                                                  save_best_only=self.settings.save_best_only,
                                                  monitor=self.settings.monitor)

    def _csv_logger(self):
        return tf.keras.callbacks.CSVLogger(filename=self.settings.training_log_path)

    def _early_stopping(self):
        return tf.keras.callbacks.EarlyStopping(monitor=self.settings.monitor,
                                                patience=self.settings.early_stopping_patience)

    def _reduce_lr_onplateu(self):
        return tf.keras.callbacks.ReduceLROnPlateau(monitor=self.settings.monitor,
                                                    factor=self.settings.reduce_lr_factor,
                                                    patience=self.settings.reduce_lr_patience,
                                                    min_lr=self.settings.reduce_lr_min_lr)
