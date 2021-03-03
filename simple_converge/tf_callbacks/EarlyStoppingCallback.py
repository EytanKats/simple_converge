import tensorflow as tf
from simple_converge.tf_callbacks.BaseCallback import BaseCallback


class EarlyStoppingCallback(BaseCallback):

    """
    This class encapsulates tensorflow early stopping callback
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(EarlyStoppingCallback, self).__init__()

        self.monitor = "val_loss"
        self.patience = 10

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(EarlyStoppingCallback, self).parse_args(**kwargs)

        if "patience" in self.params.keys():
            self.patience = self.params["patience"]

        if "monitor" in self.params.keys():
            self.monitor = self.params["monitor"]

    def get_callback(self):

        """
        This method returns callback
        :return: callback
        """

        callback = tf.keras.callbacks.EarlyStopping(monitor=self.monitor,
                                                    patience=self.patience)

        return callback
