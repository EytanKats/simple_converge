import tensorflow as tf
from simple_converge.tf_regularizers.BaseRegularizer import BaseRegularizer


class L2Regularizer(BaseRegularizer):
    """
    This class encapsulates tensorflow SGD optimizer
    """

    def __init__(self):
        """
        This method initializes parameters
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(L2Regularizer, self).__init__()

        self.reg_factor = 1e-2

    def parse_args(self, **kwargs):
        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(L2Regularizer, self).parse_args(**kwargs)

        if "reg_factor" in self.params.keys():
            self.reg_factor = self.params["reg_factor"]

    def get_regularizer(self):
        """
        This method returns regularizer
        :return: regularizer
        """

        regularizer = tf.keras.regularizers.l2(l=self.reg_factor)

        return regularizer
