import tensorflow as tf
from utils.constants import EPSILON
from tf_loss_functions.core.WeightedCategoricalCrossentropy import WeightedCategoricalCrossentropy


def ce(y_true, y_pred):

    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    clipped_y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    loss = -y_true * tf.math.log(clipped_y_pred)
    mean_loss = tf.reduce_mean(loss)

    return mean_loss


def bce(y_true, y_pred):

    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    clipped_y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    loss = -(y_true * tf.math.log(clipped_y_pred) + (1 - y_true) * tf.math.log(1 - clipped_y_pred))
    mean_loss = tf.reduce_mean(loss)

    return mean_loss


def binary_focal(y_true, y_pred, alpha, gamma):

    # Clip to prevent NaN's and Inf's
    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    clipped_y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    loss = -(y_true * tf.math.log(clipped_y_pred) * alpha * tf.math.pow(1. - clipped_y_pred, gamma)
               + (1 - y_true) * tf.math.log(1 - clipped_y_pred) * (1. - alpha) * tf.keras.backend.pow(clipped_y_pred, gamma))
    mean_loss = tf.reduce_mean(loss)

    return mean_loss


def categorical_focal(y_true, y_pred, alpha, gamma):

    # Clip to prevent NaN's and Inf's
    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate cross entropy
    cross_entropy = - y_true * tf.math.log(y_pred)

    # Calculate focal metric
    loss = tf.reduce_sum(alpha * tf.keras.backend.pow(1. - y_pred, gamma) * cross_entropy, axis=-1)
    mean_loss = tf.reduce_mean(loss)

    return mean_loss


def weighted_categorical_ce(self):

    """
    This method returns class that implements weighted categorical crossentropy loss.
    Different types of misclassification weighted differently according to cost matrix.
    :return: instance of WeightedCategoricalCrossentropy class
    """

    loss_class_instance = WeightedCategoricalCrossentropy(cost_mat=self.cost_matrix)
    return loss_class_instance
