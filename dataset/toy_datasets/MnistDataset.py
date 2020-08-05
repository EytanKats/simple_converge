import os
import cv2
import numpy as np
import pandas as pd
from sklearn import metrics as skl_metrics


from plots import plots
from metrics import metrics
from utils.RunMode import RunMode
from utils.Array import Array as ArrayUtils
from dataset.BaseClassesDataset import BaseClassesDataset


class Dataset(BaseClassesDataset):

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(Dataset, self).__init__()

        # Fields to be filled during execution
        self.raw_test_predictions = None

        self.accuracies = list()
        self.confident_accuracies = list()

        self.all_raw_test_predictions = None  # raw predictions for all folds
        self.all_postprocessed_test_predictions = None  # postprocessed predictions for all folds
        self.all_true_labels = None  # true labels for all folds

        self.confusion_matrix = None
        self.confident_confusion_matrix = None

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Dataset, self).parse_args(**kwargs)

    def initialize_dataset(self):

        super(Dataset, self).initialize_dataset()

        # Initialize shape of the fields that to be filled during execution
        self.all_raw_test_predictions = np.array([]).reshape(0, len(self.class_labels))  # raw predictions for all folds
        self.all_postprocessed_test_predictions = np.array([]).reshape(0, len(self.class_labels))  # postprocessed predictions for all folds
        self.all_true_labels = np.array([]).reshape(0, len(self.class_labels))  # true labels for all folds

        self.confusion_matrix = np.zeros(shape=(len(self.class_labels), len(self.class_labels)), dtype=np.int)
        self.confident_confusion_matrix = np.zeros(shape=(len(self.class_labels), len(self.class_labels)), dtype=np.int)

    def _get_data(self, info_row):

        path = info_row[self.data_path_column]
        data = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)

        return data

    def _get_label(self, info_row):

        label = np.zeros(shape=(len(self.class_labels),))
        label[info_row[self.label_column]] = 1

        return label

    def _apply_augmentations(self, data, label):

        return data, label

    def _apply_preprocessing(self, data, label, info_row, run_mode=RunMode.TRAINING):

        return data, label

    def apply_postprocessing(self, test_predictions, test_data, original_test_data,
                             data_info, fold_num, run_mode=RunMode.TRAINING):

        # Save raw predictions
        self.raw_test_predictions = test_predictions
        self.all_raw_test_predictions = np.concatenate((self.all_raw_test_predictions, self.raw_test_predictions))

        # Classify according to maximum output value or apply threshold on most important class
        postprocessed_predictions = list()
        for prediction in self.raw_test_predictions:
            postprocessed_prediction = np.zeros(shape=(len(self.class_labels),))
            postprocessed_prediction[np.argmax(prediction)] = 1
            postprocessed_predictions.append(postprocessed_prediction)

        # Save postprocessed predictions
        self.all_postprocessed_test_predictions = np.concatenate((self.all_postprocessed_test_predictions, postprocessed_predictions))

        return np.array(postprocessed_predictions)

    def calculate_fold_metrics(self, test_predictions, test_data, original_test_data,
                               fold_test_info, fold_num, output_folder):

        true_labels = test_data[1]
        self.all_true_labels = np.concatenate((self.all_true_labels, true_labels))

        # Remove not confident predictions
        confident_test_predictions = np.copy(test_predictions)
        confident_true_labels = np.copy(true_labels)
        not_confident_idxs = list()
        for idx in range(len(self.raw_test_predictions)):
            logit = np.argmax(test_predictions[idx])
            if self.raw_test_predictions[idx][logit] < self.confidence_level:
                not_confident_idxs.append(idx)

        confident_test_predictions = np.delete(confident_test_predictions, not_confident_idxs, axis=0)
        confident_true_labels = np.delete(confident_true_labels, not_confident_idxs, axis=0)

        accuracy = skl_metrics.accuracy_score(true_labels, test_predictions)
        confident_accuracy = skl_metrics.accuracy_score(confident_true_labels, confident_test_predictions)

        confusion_matrix = skl_metrics.confusion_matrix(np.argmax(true_labels, axis=1), np.argmax(test_predictions, axis=1))
        normalized_confusion_matrix = metrics.normalized_confusion_matrix(confusion_matrix)
        confident_confusion_matrix = skl_metrics.confusion_matrix(np.argmax(confident_true_labels, axis=1), np.argmax(confident_test_predictions, axis=1))
        normalized_confident_confusion_matrix = metrics.normalized_confusion_matrix(confident_confusion_matrix)

        self.accuracies.append(accuracy)
        self.confident_accuracies.append(confident_accuracy)

        self.confusion_matrix += confusion_matrix
        self.confident_confusion_matrix += confident_confusion_matrix

        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix.png"), confusion_matrix, self.class_names)
        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_normalized.png"), confusion_matrix, self.class_names, normalize=True)
        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_confident.png"), confident_confusion_matrix, self.class_names)
        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_confident_normalized.png"), confident_confusion_matrix, self.class_names, normalize=True)

        self.logger.log("Accuracy = {0:.1f}%".format(accuracy * 100))
        self.logger.log("Confident accuracy = {0:.1f}%".format(confident_accuracy * 100))
        self.logger.log("Confusion matrix = \n{0}".format(confusion_matrix))
        self.logger.log("Normalized confusion matrix = \n{0}".format(ArrayUtils.format_array(normalized_confusion_matrix, ".2f")))
        self.logger.log("Confident confusion matrix = \n{0}".format(confident_confusion_matrix))
        self.logger.log("Normalized confident confusion matrix = \n{0}".format(ArrayUtils.format_array(normalized_confident_confusion_matrix, ".2f")))

        return accuracy

    def save_tested_data(self, test_predictions, test_data, original_test_data,
                         fold_test_info, fold_num, output_folder):

        # Save dataframe
        paths = list()
        class_predictions = list()
        class_names = list()
        for prediction_idx, prediction in enumerate(test_predictions):

            class_label = np.argmax(prediction)  # get predicted class label

            info_row = fold_test_info.iloc[prediction_idx]
            paths.append(info_row[self.data_path_column])
            class_predictions.append(self.raw_test_predictions[prediction_idx][class_label])
            class_names.append(self.class_names[class_label])

        df = pd.DataFrame()
        df[self.data_path_column] = paths
        df["class_name"] = class_names
        df["class_prediction"] = class_predictions
        df.to_csv(os.path.join(output_folder, "predictions.csv"))

        # Save images
        true_labels = test_data[1]
        for prediction_idx, prediction in enumerate(test_predictions):

            info_row = fold_test_info.iloc[prediction_idx]
            img_path = info_row[self.data_path_column]
            img_name = os.path.basename(img_path)

            true_predictions_folder = os.path.join(output_folder, self.true_predictions_folder)
            if not os.path.exists(true_predictions_folder):
                os.makedirs(true_predictions_folder)

            false_predictions_folder = os.path.join(output_folder, self.false_predictions_folder)
            if not os.path.exists(false_predictions_folder):
                os.makedirs(false_predictions_folder)

            title = "prediction = {0}\n".format(self.raw_test_predictions[prediction_idx])
            if np.argmax(prediction) == np.argmax(true_labels[prediction_idx]) and self.save_true_predictions:
                output_dir = os.path.join(true_predictions_folder, img_name)
                img = cv2.cvtColor(original_test_data[0][prediction_idx], cv2.COLOR_BGR2RGB)
                plots.image_plot(img, title=title, output_path=output_dir)

            elif np.argmax(prediction) == np.argmax(true_labels[prediction_idx]) \
                    and self.save_not_confident_true_predictions \
                    and np.max(self.raw_test_predictions[prediction_idx]) < self.save_confidence_thr:
                output_dir = os.path.join(true_predictions_folder, img_name)
                img = cv2.cvtColor(original_test_data[0][prediction_idx], cv2.COLOR_BGR2RGB)
                plots.image_plot(img, title=title, output_path=output_dir)

            elif np.argmax(prediction) != np.argmax(true_labels[prediction_idx]) \
                    and self.save_false_predictions:
                output_dir = os.path.join(false_predictions_folder, img_name)
                img = cv2.cvtColor(original_test_data[0][prediction_idx], cv2.COLOR_BGR2RGB)
                plots.image_plot(img, title=title, output_path=output_dir)

    def save_inferenced_data(self, inference_predictions, inference_data, original_inference_data,
                             batch_inference_info, batch_num, output_folder):

        pass


    def log_metrics(self, output_folder):

        self.logger.log("\n\nOverall metrics:")

        self.logger.log("\nAccuracy mean = {0:.1f}%".format(np.mean(self.accuracies) * 100))
        self.logger.log("Accuracy std = {0:.1f}%".format(np.std(self.accuracies) * 100))

        self.logger.log("\nConfident accuracy mean = {0:.1f}%".format(np.mean(self.confident_accuracies) * 100))
        self.logger.log("Confident accuracy std = {0:.1f}%".format(np.std(self.confident_accuracies) * 100))

        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix.png"), self.confusion_matrix, self.class_names)
        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_normalized.png"), self.confusion_matrix, self.class_names, normalize=True)
        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_confident.png"), self.confident_confusion_matrix, self.class_names)
        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_confident_normalized.png"), self.confident_confusion_matrix, self.class_names, normalize=True)

        normalized_confusion_matrix = metrics.normalized_confusion_matrix(self.confusion_matrix)
        normalized_confident_confusion_matrix = metrics.normalized_confusion_matrix(self.confident_confusion_matrix)

        self.logger.log("\nConfusion matrix = \n{0}".format(self.confusion_matrix))
        self.logger.log("Normalized confusion matrix = \n{0}".format(ArrayUtils.format_array(normalized_confusion_matrix, ".2f")))
        self.logger.log("\nConfident confusion matrix = \n{0}".format(self.confident_confusion_matrix))
        self.logger.log("Normalized confident confusion matrix = \n{0}".format(ArrayUtils.format_array(normalized_confident_confusion_matrix, ".2f")))

        # Calculate optimal threshold based on ROC (max(tpr + (1-fpr))) and plot ROC
        all_true_labels = [np.argmax(label) for label in self.all_true_labels]
        all_raw_test_predictions = [prediction[1] for prediction in self.all_raw_test_predictions]

        fpr, tpr, thr = skl_metrics.roc_curve(all_true_labels, all_raw_test_predictions)
        plots.roc_plot(os.path.join(output_folder, "roc_{0}_vs_others".format(self.class_names[1])), fpr, tpr, thr)

        optimal_thr = thr[np.argmax(np.add(1 - fpr, tpr))]
        self.logger.log("\nOptimal threshold for {0} classification = {1:.2f}".format(self.class_names[1], optimal_thr))

        # Calculate optimal confusion matrix
        postprocessed_prediction = np.zeros_like(self.all_raw_test_predictions)
        for prediction_idx, prediction in enumerate(self.all_raw_test_predictions):
            if prediction[0] >= optimal_thr:
                postprocessed_prediction[prediction_idx][0] = 1
            else:
                postprocessed_prediction[prediction_idx][1] = 1

        optimal_confusion_matrix = skl_metrics.confusion_matrix(np.argmax(self.all_true_labels, axis=1), np.argmax(postprocessed_prediction, axis=1))
        optimal_normalized_confusion_matrix = metrics.normalized_confusion_matrix(optimal_confusion_matrix)

        optimal_accuracy = skl_metrics.accuracy_score(self.all_true_labels, postprocessed_prediction)

        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_optimal.png"), optimal_confusion_matrix, self.class_names)
        plots.confusion_matrix_plot(os.path.join(output_folder, "confusion_matrix_optimal_normalized.png"), optimal_confusion_matrix, self.class_names, normalize=True)

        self.logger.log("\nOptimal accuracy = {0}".format(optimal_accuracy))
        self.logger.log("Optimal confusion matrix = \n{0}".format(optimal_confusion_matrix))
        self.logger.log("Optimal normalized confusion matrix = \n{0}".format(ArrayUtils.format_array(optimal_normalized_confusion_matrix, ".2f")))
