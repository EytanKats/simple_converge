import tensorflow as tf
from utils.constants import EPSILON


def precision(y_true, y_pred, batch_wise):

    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    precision_numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
    precision_denominator = tf.reduce_sum(y_pred, axis=(1, 2, 3))

    if batch_wise:
        precision_score = (tf.reduce_sum(precision_numerator) + epsilon) / (tf.reduce_sum(precision_denominator) + epsilon)
    else:
        slice_precision_score = (precision_numerator + epsilon) / (precision_denominator + epsilon)
        precision_score = tf.reduce_mean(slice_precision_score)

    return precision_score


def recall(y_true, y_pred, batch_wise):

    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    recall_numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
    recall_denominator = tf.reduce_sum(y_true, axis=(1, 2, 3))

    if batch_wise:
        recall_score = (tf.reduce_sum(recall_numerator) + epsilon) / (tf.reduce_sum(recall_denominator) + epsilon)
    else:
        slice_recall_score = (recall_numerator + epsilon) / (recall_denominator + epsilon)
        recall_score = tf.reduce_mean(slice_recall_score)

    return recall_score
