import tensorflow as tf
from simple_converge.tf_metrics.BaseMetric import BaseMetric


class DifferenceMetric(BaseMetric):

    """
    This class implements metrics that measures differences between images:
     - mean absolute error
     - mean squared error
     - structural similarity
     - multi-scale structural similarity
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(DifferenceMetric, self).__init__()

        self.loss = False

        self.mean_absolute_error = False
        self.mean_squared_error = False
        self.ssim = False
        self.ms_ssim = False

        self.ms_ssim_power_factors = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(DifferenceMetric, self).parse_args(**kwargs)

        if "loss" in self.params.keys():
            self.loss = self.params["loss"]

        if "mean_absolute_error" in self.params.keys():
            self.mean_absolute_error = self.params["mean_absolute_error"]

        if "mean_squared_error" in self.params.keys():
            self.mean_squared_error = self.params["mean_squared_error"]

        if "ssim" in self.params.keys():
            self.ssim = self.params["ssim"]

        if "ms_ssim" in self.params.keys():
            self.ms_ssim = self.params["ms_ssim"]

        if "ms_ssim_power_factors" in self.params.keys():
            self.ms_ssim_power_factors = self.params["ms_ssim_power_factors"]

    def get_metric(self):

        """
        This method returns metric function according to parameters
        :return: metric function
        """

        if self.mean_absolute_error:
            return self.mean_absolute_error_metric

        elif self.mean_squared_error:
            return self.mean_squared_error_metric

        elif self.ssim:
            return self.ssim_metric

        elif self.ms_ssim:
            return self.ms_ssim_metric

        else:
            raise ValueError("The metric is not found")

    @staticmethod
    def mean_absolute_error_metric(y_true, y_pred):

        """
        This method calculates mean absolute error
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: mean absolute error
        """

        metric = tf.keras.losses.mean_absolute_error(y_true, y_pred)

        return metric
    
    @staticmethod
    def mean_squared_error_metric(y_true, y_pred):

        """
        This method calculates mean squared error
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: mean squared error
        """

        metric = tf.keras.losses.mean_squared_error(y_true, y_pred)

        return metric

    def ssim_metric(self, y_true, y_pred):

        """
        This method calculates structural similarity metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: structural similarity metric
        """

        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        if self.loss:
            res = 1 - ssim
        else:
            res = ssim

        return res

    def ms_ssim_metric(self, y_true, y_pred):

        """
        This method calculates calculates multi-scale structural similarity metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: multi-scale structural similarity metric
        """

        ssim = tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, power_factors=self.ms_ssim_power_factors, max_val=1.0))

        if self.loss:
            res = 1 - ssim
        else:
            res = ssim

        return res
