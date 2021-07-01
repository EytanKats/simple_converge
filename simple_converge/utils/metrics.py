import numpy as np
from simple_converge.utils.constants import EPSILON


def accuracy(predictions, ground_truth):

    """
    This method calculates accuracy metric between predictions and ground truth values
    :param predictions: predictions (1D array)
    :param ground_truth: ground truth values (1D array)
    :return: accuracy score
    """

    true_predicted_classes = np.equal(predictions, ground_truth)
    _accuracy = np.sum(true_predicted_classes) / len(ground_truth)

    return _accuracy


def kappa(predictions, ground_truth):

    """
    This method calculates kappa metric https://en.wikipedia.org/wiki/Cohen%27s_kappa
     between predictions and ground truth values
    :param predictions: predictions (1D array)
    :param ground_truth: ground truth values (1D array)
    :return: kappa score
    """

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


def iou(predictions, ground_truth):

    """
    This method calculates iou (jaccard) metric between predictions and ground truth
    :param predictions: model predictions
    :param ground_truth: ground truth
    :return: iou score
    """

    numerator = np.sum(predictions * ground_truth)
    denominator = np.sum(predictions) + np.sum(ground_truth) - numerator
    iou_score = (numerator + EPSILON) / (denominator + EPSILON)

    return iou_score


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


def contrast_ratio(target, background):

    """
    This method calculates contrast ration between target and background image regions
    :param target: target image region
    :param background: background image region
    :return: contrast ration
    """

    mean_target = np.mean(target)
    mean_background = np.mean(background)

    ratio = 20 * np.log10(mean_target / mean_background)

    return ratio


def contrast_to_noise_ratio(target, background):

    """
    This method calculates contrast-to-noise ratio using target and background image regions
    :param target: target image region
    :param background: background image region
    :return: contrast-to-noise ration
    """

    mean_target = np.mean(target)
    std_target = np.std(target)

    mean_background = np.mean(background)
    std_background = np.std(background)

    numerator = mean_target - mean_background
    denominator = np.sqrt(np.power(std_target, 2) + np.power(std_background, 2))

    ratio = numerator / denominator

    return ratio


def signal_to_noise_ratio(data):

    """
    This method calculates signal-to-noise ratio
    :param data: image region based on which signal-to-noise ratio is calculated
    :return: signal-to-noise ratio
    """

    mean_target = np.mean(data)
    std_target = np.std(data)

    ratio = mean_target / std_target

    return ratio


def normalized_confusion_matrix(confusion_matrix):

    """
    This method normalizes confusion matrix
    :param confusion_matrix: confusion matrix (2D array)
    :return: normalized confusion matrix
    """

    _normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    return _normalized_confusion_matrix
