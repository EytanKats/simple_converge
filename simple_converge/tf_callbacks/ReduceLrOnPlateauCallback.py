import tensorflow as tf
from simple_converge.tf_callbacks.BaseCallback import BaseCallback


class ReduceLrOnPlateauCallback(BaseCallback):

    """
    This class encapsulates tensorflow reduce learning rate on plateau callback
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(ReduceLrOnPlateauCallback, self).__init__()

        self.monitor = "val_loss"
        self.reduce_factor = 0.9
        self.patience = 3
        self.min_lr = 1e-5

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(ReduceLrOnPlateauCallback, self).parse_args(**kwargs)

        if "monitor" in self.params.keys():
            self.monitor = self.params["monitor"]

        if "reduce_factor" in self.params.keys():
            self.reduce_factor = self.params["reduce_factor"]

        if "patience" in self.params.keys():
            self.patience = self.params["patience"]

        if "min_lr" in self.params.keys():
            self.min_lr = self.params["min_lr"]

    def get_callback(self):

        """
        This method returns callback
        :return: callback
        """

        callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=self.monitor,
                                                         factor=self.reduce_factor,
                                                         patience=self.patience,
                                                         min_lr=self.min_lr)

        return callback
