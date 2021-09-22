import tensorflow as tf
from simple_converge.utils.constants import EPSILON
from simple_converge.tf_metrics.BaseMetric import BaseMetric


class SegmentationMetric(BaseMetric):

    """
    This class implements metrics that measures differences between masks
     - dice
     - focal dice
     - tversky
     - focal tversky
     - precision
     - recall
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(SegmentationMetric, self).__init__()

        self.loss = False

        self.dice = False
        self.focal_dice = False
        self.focal_tversky = False
        self.tversky = False
        self.precision = False
        self.recall = False

        self.batch_wise = True

        self.fp_coeff = 0.5
        self.fn_coeff = 0.5

        self.focal_gamma = 2

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(SegmentationMetric, self).parse_args(**kwargs)

        if "loss" in self.params.keys():
            self.loss = self.params["loss"]

        if "dice" in self.params.keys():
            self.dice = self.params["dice"]

        if "focal_dice" in self.params.keys():
            self.focal_dice = self.params["focal_dice"]

        if "tversky" in self.params.keys():
            self.tversky = self.params["tversky"]

        if "focal_tversky" in self.params.keys():
            self.focal_tversky = self.params["focal_tversky"]

        if "precision" in self.params.keys():
            self.precision = self.params["precision"]

        if "recall" in self.params.keys():
            self.recall = self.params["recall"]

        if "batch_wise" in self.params.keys():
            self.batch_wise = self.params["batch_wise"]

        if "fp_coeff" in self.params.keys():
            self.fp_coeff = self.params["fp_coeff"]

        if "fn_coeff" in self.params.keys():
            self.fn_coeff = self.params["fn_coeff"]

        if "focal_gamma" in self.params.keys():
            self.focal_gamma = self.params["focal_gamma"]

    def get_metric(self):

        """
        This method returns metric function according to parameters
        :return: metric function
        """

        if self.dice:
            return self.dice_metric

        elif self.focal_dice:
            return self.focal_dice_metric

        elif self.tversky:
            return self.tversky_metric

        elif self.focal_tversky:
            return self.focal_tversky_metric

        elif self.precision:
            return self.precision_metric

        elif self.recall:
            return self.recall_metric

        else:
            raise ValueError("The metric is not found")

    def dice_metric(self, y_true, y_pred):

        """
        This method calculates dice metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: dice metric
        """

        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        dice_numerator = 2.0 * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
        dice_denominator = tf.reduce_sum(y_pred, axis=(1, 2, 3)) + tf.reduce_sum(y_true, axis=(1, 2, 3))

        if self.batch_wise:
            dice_score = (tf.reduce_sum(dice_numerator) + epsilon) / (tf.reduce_sum(dice_denominator) + epsilon)
        else:
            slice_dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
            dice_score = tf.reduce_mean(slice_dice_score)

        if self.loss:
            res = 1 - dice_score
        else:
            res = dice_score

        return res

    def focal_dice_metric(self, y_true, y_pred):

        """
        This method calculates focal dice metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: focal dice metric
        """

        dice_metric = self.dice_metric(y_true, y_pred)
        metric = tf.math.pow(dice_metric, self.focal_gamma)

        return metric

    def tversky_metric(self, y_true, y_pred):

        """
        This method calculates tversky metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: metric function
        """

        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
        false_positives = tf.reduce_sum(y_pred, axis=(1, 2, 3)) - intersection
        false_negatives = tf.reduce_sum(y_true, axis=(1, 2, 3)) - intersection
        tversky_denominator = intersection + self.fp_coeff * false_positives + self.fn_coeff * false_negatives

        if self.batch_wise:
            tversky_score = (tf.reduce_sum(intersection) + epsilon) / (tf.reduce_sum(tversky_denominator) + epsilon)
        else:
            slice_tversky_score = (intersection + epsilon) / (tversky_denominator + epsilon)
            tversky_score = tf.reduce_mean(slice_tversky_score)

        if self.loss:
            res = 1 - tversky_score
        else:
            res = tversky_score

        return res

    def focal_tversky_metric(self, y_true, y_pred):

        """
        This method calculates focal tversky metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: focal tversky metric
        """

        tversky_metric = self.tversky_metric(y_true, y_pred)
        metric = tf.math.pow(tversky_metric, self.focal_gamma)

        return metric

    def precision_metric(self, y_true, y_pred):

        """
        This method calculates dice metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: precision metric
        """

        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        precision_numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
        precision_denominator = tf.reduce_sum(y_pred, axis=(1, 2, 3))

        if self.batch_wise:
            precision_score = (tf.reduce_sum(precision_numerator) + epsilon) / (tf.reduce_sum(precision_denominator) + epsilon)
        else:
            slice_precision_score = (precision_numerator + epsilon) / (precision_denominator + epsilon)
            precision_score = tf.reduce_mean(slice_precision_score)

        return precision_score

    def recall_metric(self, y_true, y_pred):

        """
        This method calculates recall metric
        :param y_true: ground truth labels
        :param y_pred: predictions
        :return: recall metric
        """

        epsilon = tf.constant(EPSILON, dtype=tf.float32)
        recall_numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
        recall_denominator = tf.reduce_sum(y_true, axis=(1, 2, 3))

        if self.batch_wise:
            recall_score = (tf.reduce_sum(recall_numerator) + epsilon) / (tf.reduce_sum(recall_denominator) + epsilon)
        else:
            slice_recall_score = (recall_numerator + epsilon) / (recall_denominator + epsilon)
            recall_score = tf.reduce_mean(slice_recall_score)

        return recall_score
