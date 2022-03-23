import numpy as np
from sklearn import metrics as skl_metrics
from imblearn import metrics as imb_metrics


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


def categorical_classification_metrics(
        predicted_labels,
        gt_labels,
        predicted_probs
):

    auc_list = list()
    acc_list = list()
    sens_list = list()
    spec_list = list()
    prec_list = list()
    f1_list = list()

    num_of_classes = len(predicted_labels[0])
    for i in range(num_of_classes):
        auc_list.append(skl_metrics.roc_auc_score(y_true=gt_labels[:, i], y_score=predicted_probs[:, i]))
        acc_list.append(skl_metrics.accuracy_score(y_true=gt_labels[:, i], y_pred=predicted_labels[:, i]))
        sens_list.append(imb_metrics.sensitivity_score(y_true=gt_labels[:, i], y_pred=predicted_labels[:, i]))
        spec_list.append(imb_metrics.specificity_score(y_true=gt_labels[:, i], y_pred=predicted_labels[:, i]))
        prec_list.append(skl_metrics.precision_score(y_true=gt_labels[:, i], y_pred=predicted_labels[:, i]))
        f1_list.append(skl_metrics.f1_score(y_true=gt_labels[:, i], y_pred=predicted_labels[:, i]))

    res = {
        'auc': auc_list,
        'acc': acc_list,
        'sens': sens_list,
        'spec': spec_list,
        'prec': prec_list,
        'f1': f1_list
    }

    return res


def metric_vs_discarded_samples(
        metric,
        predicted_labels,
        gt_labels,
        predicted_probs,
        relevant_indices=None,  # percentage of discarded samples will be calculated relatively to this indices
        confidence_thr_list=np.linspace(0, 1, num=1001)
):

    score_list = []
    discarded_list = []
    for confidence_thr in confidence_thr_list:
        max_prob = np.max(predicted_probs, axis=1)
        not_confident_idxs = list(np.where(max_prob < confidence_thr)[0])
        confident_predicted_labels = np.delete(predicted_labels, not_confident_idxs, axis=0)
        confident_gt_labels = np.delete(gt_labels, not_confident_idxs, axis=0)

        if relevant_indices is None:
            discarded = (len(not_confident_idxs) / len(max_prob)) * 100
        else:
            relevant_max_prob = max_prob[relevant_indices]
            relevant_not_confident_idxs = list(np.where(relevant_max_prob < confidence_thr)[0])
            discarded = (len(relevant_not_confident_idxs) / len(relevant_max_prob)) * 100

        if discarded < 100:
            score = metric(confident_gt_labels, confident_predicted_labels)
        else:
            score = 1

        discarded_list.append(discarded)
        score_list.append(score)

    interp_discarded = np.linspace(0, 100, num=101)
    interp_score = np.interp(interp_discarded, discarded_list, score_list)
    interp_conf = np.interp(interp_discarded, discarded_list, confidence_thr_list)

    return interp_score, interp_conf, interp_discarded
