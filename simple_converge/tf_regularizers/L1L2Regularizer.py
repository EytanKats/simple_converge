import tensorflow as tf
from simple_converge.tf_regularizers.BaseRegularizer import BaseRegularizer


class L1L2Regularizer(BaseRegularizer):
    """
    This class encapsulates tensorflow SGD optimizer
    """

    def __init__(self):
        """
        This method initializes parameters
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(L1L2Regularizer, self).__init__()

        self.l1_reg_factor = 1e-2
        self.l2_reg_factor = 1e-2

    def parse_args(self, **kwargs):
        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(L1L2Regularizer, self).parse_args(**kwargs)

        if "l1_reg_factor" in self.params.keys():
            self.l1_reg_factor = self.params["l1_reg_factor"]

        if "l2_reg_factor" in self.params.keys():
            self.l2_reg_factor = self.params["l2_reg_factor"]

    def get_regularizer(self):
        """
        This method returns regularizer
        :return: regularizer
        """

        regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_reg_factor, l2=self.l2_reg_factor)

        return regularizer
