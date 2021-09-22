import tensorflow as tf

from simple_converge.utils.constants import EPSILON
from simple_converge.tf_metrics.BaseMetric import BaseMetric
from simple_converge.tf_metrics.core.WeightedCategoricalCrossentropy import WeightedCategoricalCrossentropy


class CrossEntropyMetric(BaseMetric):

    """
    This class implements cross entropy metrics:
     - binary cross entropy
     - categorical cross entropy
     - binary focal cross entropy
     - categorical focal cross entropy
     - weighted categorical cross entropy
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(CrossEntropyMetric, self).__init__()

        self.categorical_cross_entropy = False
        self.binary_cross_entropy = False

        self.categorical_focal = False
        self.binary_focal = False

        self.weighted_categorical_cross_entropy = False

        self.focal_gamma = 2
        self.focal_alpha = 0.25

        self.cost_matrix = None

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(CrossEntropyMetric, self).parse_args(**kwargs)

        if "categorical_cross_entropy" in self.params.keys():
            self.categorical_cross_entropy = self.params["categorical_cross_entropy"]

        if "binary_cross_entropy" in self.params.keys():
            self.binary_cross_entropy = self.params["binary_cross_entropy"]

        if "categorical_focal" in self.params.keys():
            self.categorical_focal = self.params["categorical_focal"]

        if "binary_focal" in self.params.keys():
            self.binary_focal = self.params["binary_focal"]

        if "weighted_categorical_cross_entropy" in self.params.keys():
            self.weighted_categorical_cross_entropy = self.params["weighted_categorical_cross_entropy"]

        if "focal_alpha" in self.params.keys():
            self.focal_alpha = self.params["focal_alpha"]

        if "focal_gamma" in self.params.keys():
            self.focal_gamma = self.params["focal_gamma"]

        if "cost_matrix" in self.params.keys():
            self.cost_matrix = self.params["cost_matrix"]

    def get_metric(self):

        if self.categorical_cross_entropy:
            return self.categorical_cross_entropy_metric

        elif self.binary_cross_entropy:
            return self.binary_cross_entropy_metric

        elif self.categorical_focal:
            return self.categorical_focal_metric

        elif self.binary_focal:
            return self.binary_focal_metric

        elif self.weighted_categorical_cross_entropy:
            return self.weighted_categorical_cross_entropy_metric()

        else:
            raise ValueError("The metric is not found")

    @staticmethod
    def categorical_cross_entropy_metric(y_true, y_pred):

        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        clipped_y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        metric = -y_true * tf.math.log(clipped_y_pred)
        mean_metric = tf.reduce_mean(metric)

        return mean_metric

    @staticmethod
    def binary_cross_entropy_metric(y_true, y_pred):

        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        clipped_y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        metric = -(y_true * tf.math.log(clipped_y_pred) + (1 - y_true) * tf.math.log(1 - clipped_y_pred))
        mean_metric = tf.reduce_mean(metric)

        return mean_metric

    def binary_focal_metric(self, y_true, y_pred):

        """
        This method calculates binary focal cross entropy
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: metric function
        """

        # Clip to prevent NaN's and Inf's
        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        clipped_y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        metric = -(y_true * tf.math.log(clipped_y_pred) * self.focal_alpha * tf.math.pow(1. - clipped_y_pred, self.focal_gamma)
                   + (1 - y_true) * tf.math.log(1 - clipped_y_pred) * (1. - self.focal_alpha) * tf.keras.backend.pow(clipped_y_pred, self.focal_gamma))
        mean_metric = tf.reduce_mean(metric)

        return mean_metric

    def categorical_focal_metric(self, y_true, y_pred):

        """
        This method calculates categorical focal cross entropy
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: metric function
        """

        # Clip to prevent NaN's and Inf's
        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate cross entropy
        cross_entropy = - y_true * tf.math.log(y_pred)

        # Calculate focal metric
        metric = tf.reduce_sum(self.focal_alpha * tf.keras.backend.pow(1. - y_pred, self.focal_gamma) * cross_entropy, axis=-1)
        mean_metric = tf.reduce_mean(metric)

        return mean_metric

    def weighted_categorical_cross_entropy_metric(self):

        """
        This method returns class that implements weighted categorical crossentropy loss.
        Different types of misclassification weighted differently according to cost matrix.
        :return: instance of WeightedCategoricalCrossentropy class
        """

        loss_class_instance = WeightedCategoricalCrossentropy(cost_mat=self.cost_matrix)
        return loss_class_instance
