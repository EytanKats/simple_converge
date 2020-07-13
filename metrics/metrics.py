import numpy as np

from utils.constants import EPSILON


def accuracy(predictions, ground_truth):

    true_predicted_classes = np.equal(predictions, ground_truth)
    _accuracy = np.sum(true_predicted_classes) / len(ground_truth)

    return _accuracy


def kappa(predictions, ground_truth):

    agreement = accuracy(predictions, ground_truth)

    # Calculate random agreement
    random_agreement = 0
    unique_prediction_classes = np.unique(predictions)
    for unique_prediction in unique_prediction_classes:
        predictions_num_for_class = np.sum(predictions == unique_prediction) / len(predictions)
        ground_truth_num_for_class = np.sum(ground_truth == unique_prediction) / len(ground_truth)
        random_agreement = random_agreement + predictions_num_for_class * ground_truth_num_for_class

    if agreement <= random_agreement:
        _kappa = 0
    else:
        _kappa = (agreement - random_agreement + np.finfo(float).eps) / (1 - random_agreement + np.finfo(float).eps)

    return _kappa


def normalized_confusion_matrix(confusion_matrix):

    _normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    return _normalized_confusion_matrix


def dice(predictions, ground_truth):

    """
    This method calculates dice metric between predictions and ground truth
    :param predictions: model predictions
    :param ground_truth: ground truth
    :return: dice score
    """

    numerator = 2.0 * np.sum(predictions * ground_truth)
    denominator = np.sum(predictions) + np.sum(ground_truth)
    dice_score = (numerator + EPSILON) / (denominator + EPSILON)

    return dice_score


def recall(predictions, ground_truth):

    """
    This method calculates recall metric between predictions and ground truth
    :param predictions: model predictions
    :param ground_truth: ground truth
    :return: recall score
    """

    numerator = np.sum(predictions * ground_truth)
    denominator = np.sum(ground_truth)
    recall_score = (numerator + EPSILON) / (denominator + EPSILON)

    return recall_score


def precision(predictions, ground_truth):

    """
    This method calculates precision metric between predictions and ground truth
    :param predictions: model predictions
    :param ground_truth: ground truth
    :return: precision score
    """

    numerator = np.sum(predictions * ground_truth)
    denominator = np.sum(predictions)
    precision_score = (numerator + EPSILON) / (denominator + EPSILON)

    return precision_score
