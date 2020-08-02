import abc
import uuid
import numpy as np

from base.BaseObject import BaseObject
from utils import dataset_utils
from utils.RunMode import RunMode


class BaseDataset(BaseObject):

    """
    This abstract class defines common methods for datasets
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(BaseDataset, self).__init__()

        # Fields to be filled by parsing
        self.data_definition_file = ""
        self.data_path_column = ""
        self.filters = dict()

        self.preload_data = False
        self.preload_labels = False

        self.inference_batch_size = 16

        # Fields to be filled during execution
        self.original_info = None
        self.filtered_info = None
        self.inference_info = list()

        self.preloaded_data = dict()
        self.preloaded_labels = dict()

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseDataset, self).parse_args(**kwargs)

        if "data_definition_file" in self.params.keys():
            self.data_definition_file = self.params["data_definition_file"]

        if "data_path_column" in self.params.keys():
            self.data_path_column = self.params["data_path_column"]

        if "filters" in self.params.keys():
            self.filters = self.params["filters"]

        if "preload_data" in self.params.keys():
            self.preload_data = self.params["preload_data"]

        if "preload_labels" in self.params.keys():
            self.preload_labels = self.params["preload_labels"]

        if "inference_batch_size" in self.params.keys():
            self.inference_batch_size = self.params["inference_batch_size"]

    def initialize_dataset(self):

        self.original_info = dataset_utils.load_dataset_file(self.data_definition_file)

        # Generate unique identifiers for data samples
        uuids = [str(uuid.uuid4()) for _ in range(self.original_info.shape[0])]
        self.original_info["uuid"] = uuids

        # Apply filters on data
        self.filtered_info = dataset_utils.apply_filters(self.original_info.copy(), self.filters)

        # Preload data
        self.preloaded_data = dict()
        if self.preload_data:
            for _, info_row in self.filtered_info.iterrows():
                self.preloaded_data[info_row["uuid"]] = self._get_data(info_row)  # preloaded data will be always preprocessed

        # Preload labels
        self.preloaded_labels = dict()
        if self.preload_labels:
            for _, info_row in self.filtered_info.iterrows():
                self.preloaded_labels[info_row["uuid"]] = self._get_label(info_row)

    @abc.abstractmethod
    def _get_data(self, info_row):
        pass

    def get_data(self, info_row):

        if self.preload_data:
            data = self.preloaded_data[info_row["uuid"]]
        else:
            data = self._get_data(info_row)

        return data

    @abc.abstractmethod
    def _get_label(self, info_row):
        pass

    def get_label(self, info_row):

        if self.preload_labels:
            label = self.preloaded_labels[info_row["uuid"]]
        else:
            label = self._get_label(info_row)

        return label

    @abc.abstractmethod
    def _apply_augmentations(self, data, label):
        pass

    @abc.abstractmethod
    def _apply_preprocessing(self, data, label, info_row, run_mode=RunMode.TRAINING):
        pass

    def get_pair(self, info_row, preprocess, augment, get_data=True, get_label=True, run_mode=RunMode.TRAINING):

        data = None
        label = None

        if get_data:
            data = self.get_data(info_row)

        if get_label:
            label = self.get_label(info_row)

        if augment:
            data, label = self._apply_augmentations(data, label)

        if preprocess:
            data, label = self._apply_preprocessing(data, label, info_row, run_mode=run_mode)

        return data, label

    def split_inference_data(self):

        inference_batches_num = np.ceil(self.filtered_info.shape[0] / self.inference_batch_size)
        self.inference_info = np.array_split(self.filtered_info, inference_batches_num)

    @abc.abstractmethod
    def apply_postprocessing(self,
                             test_predictions,
                             test_data,
                             original_test_data,
                             fold_test_info,
                             fold_num,
                             run_mode=RunMode.TRAINING):
        pass

    @abc.abstractmethod
    def calculate_fold_metrics(self,
                               test_predictions,
                               test_data,
                               original_test_data,
                               fold_test_info,
                               fold_num,
                               output_folder):
        pass

    @abc.abstractmethod
    def log_metrics(self, output_folder):
        pass

    @abc.abstractmethod
    def save_tested_data(self,
                         test_predictions,
                         test_data,
                         original_test_data,
                         fold_test_info,
                         fold_num,
                         output_folder):
        pass

    @abc.abstractmethod
    def save_inferenced_data(self,
                             inference_predictions,
                             inference_data,
                             original_inference_data,
                             batch_inference_info,
                             batch_num,
                             output_folder):
        pass
