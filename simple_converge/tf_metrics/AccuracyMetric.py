import tensorflow as tf
from simple_converge.tf_metrics.BaseMetric import BaseMetric


class AccuracyMetric(BaseMetric):

    """
    This class implements accuracy related metrics:
     - categorical accuracy
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(AccuracyMetric, self).__init__()

        self.binary_accuracy = False
        self.categorical_accuracy = False

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(AccuracyMetric, self).parse_args(**kwargs)

        if "binary_accuracy" in self.params.keys():
            self.binary_accuracy = self.params["binary_accuracy"]

        if "categorical_accuracy" in self.params.keys():
            self.categorical_accuracy = self.params["categorical_accuracy"]

    def get_metric(self):

        if self.binary_accuracy:
            return self.binary_accuracy_metric()

        if self.categorical_accuracy:
            return self.categorical_accuracy_metric()

        else:
            raise ValueError("The metric is not found")

    @staticmethod
    def binary_accuracy_metric():

        accuracy = tf.keras.metrics.BinaryAccuracy()

        return accuracy

    @staticmethod
    def categorical_accuracy_metric():

        accuracy = tf.keras.metrics.CategoricalAccuracy()

        return accuracy
