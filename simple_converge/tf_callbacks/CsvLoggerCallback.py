import tensorflow as tf
from simple_converge.tf_callbacks.BaseCallback import BaseCallback


class CsvLoggerCallback(BaseCallback):

    """
    This class encapsulates tensorflow CSV logger callback
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(CsvLoggerCallback, self).__init__()

        self.training_log_path = None

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(CsvLoggerCallback, self).parse_args(**kwargs)

        if "training_log_path" in self.params.keys():
            self.training_log_path = self.params["training_log_path"]

    def get_callback(self):

        """
        This method returns callback
        :return: callback
        """

        callback = tf.keras.callbacks.CSVLogger(filename=self.training_log_path)

        return callback
