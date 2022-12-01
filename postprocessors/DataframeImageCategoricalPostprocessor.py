import os
import copy
import numpy as np
import pandas as pd
from loguru import logger
from scipy import special
from sklearn import metrics as skl_metrics

from simple_converge.postprocessors.BasePostProcessor import BasePostProcessor
from simple_converge.utils.plots_matplotlib import confusion_matrix_plot
from simple_converge.utils.plots_mlops import metric_vs_discarded_samples_plot
from simple_converge.utils.metrics import categorical_classification_metrics, metric_vs_discarded_samples


default_postprocessor_settings = {
    'labels': [],
    'activation': 'softmax',
    'per_class_classification_report': False,
    'confusion_matrix': False,
    'recall_vs_discarded_images': False,
    'per_class_f1_vs_discarded_images': False
}


class DataframeImageCategoricalPostprocessor(BasePostProcessor):

    def __init__(
            self,
            settings,
            mlops_task,
            dataframe):

        super(DataframeImageCategoricalPostprocessor, self).__init__(settings, mlops_task)

        self.dataset_df = dataframe

        self.predicted_labels_list = list()
        self.gt_labels_list = list()
        self.gt_labels_one_hot_list = list()
        self.predicted_probs_list = list()

        self.predicted_labels = None
        self.gt_labels = None
        self.gt_labels_one_hot = None
        self.predicted_probs = None

    def postprocess_predictions(
            self,
            predictions,
            data,
            run_mode
    ):

        if self.settings['activation'] == 'sigmoid':
            predictions = special.expit(predictions)
        elif self.settings['activation'] == 'softmax':
            predictions = special.softmax(predictions, axis=-1)

        predicted_labels = np.eye(len(predictions[0]))[np.argmax(predictions, axis=1)]
        gt_labels = data[1]  # There is an assumption that data[1] contains ground truth labels

        self.predicted_labels = predicted_labels
        self.gt_labels = gt_labels.numpy()
        self.gt_labels_one_hot = np.eye(len(self.settings['labels']))[gt_labels.numpy()]
        self.predicted_probs = predictions

        self.predicted_labels_list.append(predicted_labels)
        self.gt_labels_list.append(gt_labels.numpy())
        self.gt_labels_one_hot_list.append(self.gt_labels_one_hot)
        self.predicted_probs_list.append(predictions)

    def calculate_metrics(
            self,
            output_folder,
            fold,
            task
    ):

        pass

    def save_predictions(
            self,
            output_folder,
            fold,
            task,
            run_mode
    ):
        pass

    def calculate_total_metrics(
            self,
            output_folder,
            fold,
            task
    ):

        predicted_labels = np.concatenate(self.predicted_labels_list)
        gt_labels = np.concatenate(self.gt_labels_list)
        gt_labels_one_hot = np.concatenate(self.gt_labels_one_hot_list)
        predicted_probs = np.concatenate(self.predicted_probs_list)

        # Predictions file
        np.set_printoptions(precision=2)
        self.dataset_df['probs'] = predicted_probs.tolist()
        self.dataset_df['probs'] = self.dataset_df['probs'].apply(lambda x: np.around(x, decimals=2))
        predictions_file_path = os.path.join(output_folder, 'predictions.csv')
        self.dataset_df.to_csv(predictions_file_path, index=False, float_format='%.2f')

        # Overall accuracy and error rate
        accuracy = skl_metrics.accuracy_score(gt_labels, np.argmax(predicted_labels, axis=1))
        error_rate = 1 - accuracy

        logger.info(f'\nACCURACY:{accuracy * 100:.1f}%')
        logger.info(f'ERROR RATE:{error_rate * 100:.1f}%')

        accuracy_df = pd.DataFrame(index=['overall_metrics'])
        accuracy_df['accuracy'] = [accuracy * 100]
        accuracy_df['error_rate'] = [error_rate * 100]
        accuracy_file_path = os.path.join(output_folder, 'overall_metrics.csv')
        accuracy_df.to_csv(accuracy_file_path, float_format='%.1f')

        # Per class classification report
        if self.settings['per_class_classification_report']:
            metric_scores = categorical_classification_metrics(
                predicted_labels=predicted_labels,
                predicted_probs=predicted_probs,
                gt_labels=gt_labels_one_hot
            )

            metric_scores["auc"] = [x * 100 for x in metric_scores["auc"]]
            metric_scores["acc"] = [x * 100 for x in metric_scores["acc"]]
            metric_scores["sens"] = [x * 100 for x in metric_scores["sens"]]
            metric_scores["spec"] = [x * 100 for x in metric_scores["spec"]]
            metric_scores["prec"] = [x * 100 for x in metric_scores["prec"]]
            metric_scores["f1"] = [x * 100 for x in metric_scores["f1"]]

            metric_scores["auc"].append(np.mean(metric_scores["auc"]))
            metric_scores["acc"].append(np.mean(metric_scores["acc"]))
            metric_scores["sens"].append(np.mean(metric_scores["sens"]))
            metric_scores["spec"].append(np.mean(metric_scores["spec"]))
            metric_scores["prec"].append(np.mean(metric_scores["prec"]))
            metric_scores["f1"].append(np.mean(metric_scores["f1"]))

            logger.info(f'\nMEAN METRICS:')
            logger.info(f'mean auc: {metric_scores["auc"][-1]:.1f}%')
            logger.info(f'mean accuracy: {metric_scores["acc"][-1]:.1f}%')
            logger.info(f'mean sensitivity: {metric_scores["sens"][-1]:.1f}%')
            logger.info(f'mean specificity: {metric_scores["spec"][-1]:.1f}%')
            logger.info(f'mean precision: {metric_scores["prec"][-1]:.1f}%')
            logger.info(f'mean f1: {metric_scores["f1"][-1]:.1f}%')

            for label_idx, label in enumerate(self.settings['labels']):
                logger.info(f'\n{label.upper()} METRICS:')
                logger.info(f'{label} auc: {(metric_scores["auc"][label_idx]):.1f}%')
                logger.info(f'{label} accuracy: {(metric_scores["acc"][label_idx]):.1f}%')
                logger.info(f'{label} sensitivity: {(metric_scores["sens"][label_idx]):.1f}%')
                logger.info(f'{label} specificity: {(metric_scores["spec"][label_idx]):.1f}%')
                logger.info(f'{label} precision: {(metric_scores["prec"][label_idx]):.1f}%')
                logger.info(f'{label} f1: {(metric_scores["f1"][label_idx]):.1f}%')

            metrics_df = pd.DataFrame.from_dict(metric_scores, orient='index')
            metrics_df_columns = copy.deepcopy(self.settings['labels'])
            metrics_df_columns.append('mean')
            metrics_df.columns = metrics_df_columns
            metrics_file_path = os.path.join(output_folder, 'metrics.csv')
            metrics_df.to_csv(metrics_file_path, float_format='%.1f')

        # Confusion matrix
        if self.settings['confusion_matrix']:
            confusion_matrix = skl_metrics.confusion_matrix(gt_labels, np.argmax(predicted_labels, axis=1))
            rec_confusion_matrix = skl_metrics.confusion_matrix(gt_labels, np.argmax(predicted_labels, axis=1), normalize='true')
            prec_confusion_matrix = skl_metrics.confusion_matrix(gt_labels, np.argmax(predicted_labels, axis=1), normalize='pred')

            confusion_matrix_plot(
                confusion_matrix,
                self.settings["labels"],
                output_path=os.path.join(output_folder, 'confusion_matrix'),
                fig_size=(15, 10),
                mlops_task=task,
                mlops_iteration=fold
            )

            confusion_matrix_plot(
                rec_confusion_matrix,
                self.settings["labels"],
                output_path=os.path.join(output_folder, 'confusion_matrix_recall'),
                normalize=True,
                fig_size=(15, 10),
                mlops_task=task,
                mlops_iteration=fold
            )

            confusion_matrix_plot(
                prec_confusion_matrix,
                self.settings["labels"],
                output_path=os.path.join(output_folder, 'confusion_matrix_precision'),
                normalize=True,
                fig_size=(15, 10),
                mlops_task=task,
                mlops_iteration=fold
            )

            np.set_printoptions(precision=1, linewidth=1000)
            logger.info(f'confusion matrix:\n{confusion_matrix}')
            logger.info(f'recall confusion matrix:\n{rec_confusion_matrix * 100}')
            logger.info(f'precision confusion matrix:\n{prec_confusion_matrix * 100}')

        # Recall vs discarded images
        if self.settings['recall_vs_discarded_images']:
            avg_rec_y, conf_y, discarded_x = metric_vs_discarded_samples(
                metric=skl_metrics.balanced_accuracy_score,  # metric that gets two input arguments: metric(gt_labels, predicted_labels)
                predicted_labels=np.argmax(predicted_labels, axis=1),
                predicted_probs=predicted_probs,
                gt_labels=gt_labels
            )

            metric_vs_discarded_samples_plot(
                avg_rec_y,
                conf_y,
                discarded_x,
                task,
                f'average_recall_vs_discarded_images_fold{fold}',
                'avg_recall'
            )

        # Per class f1 vs discarded images
        if self.settings['per_class_f1_vs_discarded_images']:
            num_of_classes = len(predicted_labels[0])
            for cls in range(num_of_classes):
                f1_, conf_, discarded_ = metric_vs_discarded_samples(
                    metric=skl_metrics.f1_score,  # metric that gets two input arguments: metric(gt_labels, predicted_labels)
                    predicted_labels=predicted_labels[:, cls],
                    predicted_probs=predicted_probs,
                    gt_labels=gt_labels_one_hot[:, cls],
                    relevant_indices=list(np.where(gt_labels_one_hot[:, cls] == 1)[0])  # percentage of discarded samples will be calculated relatively to this indices
                )

                metric_vs_discarded_samples_plot(
                    f1_,
                    conf_,
                    discarded_,
                    task,
                    f'f1_vs_discarded_images_fold{fold}_{self.settings["labels"][cls]}',
                    'f1'
                )
