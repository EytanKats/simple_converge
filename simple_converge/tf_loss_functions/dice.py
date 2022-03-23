import tensorflow as tf
from utils.constants import EPSILON


def dice(y_true, y_pred, batch_wise):

    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    dice_numerator = 2.0 * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
    dice_denominator = tf.reduce_sum(y_pred, axis=(1, 2, 3)) + tf.reduce_sum(y_true, axis=(1, 2, 3))

    if batch_wise:
        dice_score = (tf.reduce_sum(dice_numerator) + epsilon) / (tf.reduce_sum(dice_denominator) + epsilon)
    else:
        slice_dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
        dice_score = tf.reduce_mean(slice_dice_score)

    loss = 1 - dice_score

    return loss


def focal_dice(y_true, y_pred, gamma, batch_wise):

    dice_loss = dice(y_true, y_pred, batch_wise)
    loss = tf.math.pow(dice_loss, gamma)
    return loss


def tversky(y_true, y_pred, fp_coeff, fn_coeff, batch_wise):

    epsilon = tf.constant(EPSILON, dtype=tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=(1, 2, 3))
    false_positives = tf.reduce_sum(y_pred, axis=(1, 2, 3)) - intersection
    false_negatives = tf.reduce_sum(y_true, axis=(1, 2, 3)) - intersection
    tversky_denominator = intersection + fp_coeff * false_positives + fn_coeff * false_negatives

    if batch_wise:
        tversky_score = (tf.reduce_sum(intersection) + epsilon) / (tf.reduce_sum(tversky_denominator) + epsilon)
    else:
        slice_tversky_score = (intersection + epsilon) / (tversky_denominator + epsilon)
        tversky_score = tf.reduce_mean(slice_tversky_score)

    loss = 1 - tversky_score

    return loss


def focal_tversky(y_true, y_pred, fp_coeff, fn_coeff, batch_wise, gamma):

    tversky_score = tversky(y_true, y_pred, fp_coeff, fn_coeff, batch_wise)
    loss = tf.math.pow(tversky_score, gamma)

    return loss

