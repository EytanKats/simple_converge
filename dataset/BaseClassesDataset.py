import numpy as np
import pandas as pd

from utils.RunMode import RunMode
from utils import dataset_utils
from dataset.BaseDataset import BaseDataset


class BaseClassesDataset(BaseDataset):

    """
    This abstract class defines common methods for datasets
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(BaseClassesDataset, self).__init__()

        # Fields to be filled by parsing
        self.label_column = ""
        self.apply_class_filters = True
        self.class_filters = list()
        self.class_labels = list()
        self.class_names = list()

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseClassesDataset, self).parse_args(**kwargs)

        if "label_column" in self.params.keys():
            self.label_column = self.params["label_column"]

        if "apply_class_filters" in self.params.keys():
            self.apply_class_filters = self.params["apply_class_filters"]

        if "class_filters" in self.params.keys():
            self.class_filters = self.params["class_filters"]

        if "class_labels" in self.params.keys():
            self.class_labels = self.params["class_labels"]

        if "class_names" in self.params.keys():
            self.class_names = self.params["class_names"]

    def initialize_dataset(self, run_mode=RunMode.TRAINING):

        super(BaseClassesDataset, self).initialize_dataset(run_mode)

        if self.apply_class_filters:  # during inference there is no need to apply class filters

            # Apply classes filters to data
            classes_filtered_info = list()
            for cls_idx, class_filter in enumerate(self.class_filters):
                class_filtered_info = dataset_utils.apply_filters(self.filtered_info.copy(), class_filter)

                labels = [self.class_labels[cls_idx]] * class_filtered_info.shape[0]
                class_filtered_info[self.label_column] = labels
                classes_filtered_info.append(class_filtered_info)

                self.logger.log("{0} samples of '{1}' class".format(class_filtered_info.shape[0], self.class_names[cls_idx]))

            self.logger.log("\n")

            if run_mode == RunMode.TRAINING:
                self.filtered_info = classes_filtered_info  # filtered dataset consists of list of datasets (one for each class)
            elif run_mode == RunMode.TEST:
                self.filtered_info = pd.concat(classes_filtered_info, ignore_index=True)

                # Workaround: pd.concat changes int to float if one of the data frames is empty
                self.filtered_info[self.label_column] = self.filtered_info[self.label_column].astype(np.int32)