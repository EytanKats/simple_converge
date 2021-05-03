import tensorflow as tf
from simple_converge.tf_optimizers.BaseOptimizer import BaseOptimizer


class AdamOptimizer(BaseOptimizer):

    """
    This class encapsulates tensorflow Adam optimizer
    """

    def __init__(self):

        """
        This method initializes parameters
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(AdamOptimizer, self).__init__()

        self.learning_rate = 1e-3

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(AdamOptimizer, self).parse_args(**kwargs)

        if "learning_rate" in self.params.keys():
            self.learning_rate = self.params["learning_rate"]

    def get_optimizer(self):

        """
        This method returns optimizer
        :return: optimizer
        """

        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

        return optimizer
