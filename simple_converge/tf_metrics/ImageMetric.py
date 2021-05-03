import tensorflow as tf
from simple_converge.tf_metrics.BaseMetric import BaseMetric


class ImageMetric(BaseMetric):

    """
    This class implements variants of image relative metrics:
     - mean absolute error
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(ImageMetric, self).__init__()

        self.total_variation = False

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(ImageMetric, self).parse_args(**kwargs)

        if "total_variation" in self.params.keys():
            self.total_variation = self.params["total_variation"]

    def get_metric(self):

        """
        This method returns metric function according to parameters
        :return: metric function
        """

        if self.total_variation:
            return self.total_variation_metric

        else:
            raise ValueError("The metric is not found")

    @staticmethod
    def total_variation_metric(y_true, y_pred):

        """
        This method calculates total variation metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: total variation metric
        """

        metric = tf.reduce_sum(tf.image.total_variation(y_pred))

        return metric


