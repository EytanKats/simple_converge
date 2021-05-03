import tensorflow as tf
from simple_converge.tf_callbacks.BaseCallback import BaseCallback


class CheckpointCallback(BaseCallback):

    """
    This class encapsulates tensorflow Checkpoint callback
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(CheckpointCallback, self).__init__()

        self.checkpoint_path = None
        self.save_best_only = True
        self.save_weights_only = True
        self.monitor = "val_loss"

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(CheckpointCallback, self).parse_args(**kwargs)

        if "checkpoint_path" in self.params.keys():
            self.checkpoint_path = self.params["checkpoint_path"]

        if "save_best_only" in self.params.keys():
            self.save_best_only = self.params["save_best_only"]

        if "save_weights_only" in self.params.keys():
            self.save_weights_only = self.params["save_weights_only"]

        if "monitor" in self.params.keys():
            self.monitor = self.params["monitor"]

    def get_callback(self):

        """
        This method returns callback
        :return: callback
        """

        callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                      save_best_only=self.save_best_only,
                                                      save_weights_only=self.save_weights_only,
                                                      monitor=self.monitor)

        return callback
