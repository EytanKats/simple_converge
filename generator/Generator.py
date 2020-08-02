import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from utils.RunMode import RunMode

from base.BaseObject import BaseObject
from tf_sequences.Sequence import Sequence


class Generator(BaseObject):

    """
    This class defines data generator
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(Generator, self).__init__()

        # Fields to be filled by parsing
        self.dataset = None

        self.data_random_seed = 2018

        self.folds_num = 1

        self.data_info_folder = ""
        self.train_data_file_name = "train_data.json"
        self.val_data_file_name = "val_data.json"
        self.test_data_file_name = "test_data.json"

        self.sample_training_info = False
        self.training_data_rows = 0

        self.train_split = 0.8
        self.test_split = 0.2

        self.leave_out = False  # allows to choose for test data with unique values of 'self.leave_out_param'
        self.leave_out_param = ""
        self.leave_out_values = list()

        self.set_info = False  # set training, validation and test data info
        self.set_test_data_info = False  # set only test data info
        self.set_test_data_param = ""

        self.sequence_args = dict()

        # Fields to be filled during execution
        self.train_info = None
        self.val_info = None
        self.test_info = None

        self.leave_out_folds_values = None

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Generator, self).parse_args(**kwargs)

        if "dataset" in self.params.keys():
            self.dataset = self.params["dataset"]

        if "data_random_seed" in self.params.keys():
            self.data_random_seed = self.params["data_random_seed"]

        if "folds_num" in self.params.keys():
            self.folds_num = self.params["folds_num"]

        if "data_info_folder" in self.params.keys():
            self.data_info_folder = self.params["data_info_folder"]

        if "train_data_file_name" in self.params.keys():
            self.train_data_file_name = self.params["train_data_file_name"]

        if "val_data_file_name" in self.params.keys():
            self.val_data_file_name = self.params["val_data_file_name"]

        if "test_data_file_name" in self.params.keys():
            self.test_data_file_name = self.params["test_data_file_name"]

        if "sample_training_info" in self.params.keys():
            self.sample_training_info = self.params["sample_training_info"]

        if "training_data_rows" in self.params.keys():
            self.training_data_rows = self.params["training_data_rows"]

        if "train_split" in self.params.keys():
            self.train_split = self.params["train_split"]

        if "test_split" in self.params.keys():
            self.test_split = self.params["test_split"]

        if "leave_out" in self.params.keys():
            self.leave_out = self.params["leave_out"]

        if "leave_out_param" in self.params.keys():
            self.leave_out_param = self.params["leave_out_param"]

        if "leave_out_values" in self.params.keys():
            self.leave_out_values = self.params["leave_out_values"]

        if "set_info" in self.params.keys():
            self.set_info = self.params["set_info"]

        if "set_test_data_info" in self.params.keys():
            self.set_test_data_info = self.params["set_test_data_info"]

        if "set_test_data_param" in self.params.keys():
            self.set_test_data_param = self.params["set_test_data_param"]

        if "sequence_args" in self.params.keys():
            self.sequence_args = self.params["sequence_args"]

    def set_train_info(self):
        self.train_info = [None] * self.folds_num

        for fold in range(self.folds_num):
            fold_data_info_folder = os.path.join(self.data_info_folder, str(fold))

            train_info = pd.read_json(os.path.join(fold_data_info_folder, self.train_data_file_name))

            # randomly choose number of samples from training data
            if self.sample_training_info:
                train_info = train_info.sample(n=self.training_data_rows)

            self.train_info[fold] = train_info

    def set_val_info(self):
        self.val_info = [None] * self.folds_num

        for fold in range(self.folds_num):
            fold_data_info_folder = os.path.join(self.data_info_folder, str(fold))

            val_info = pd.read_json(os.path.join(fold_data_info_folder, self.val_data_file_name))
            self.val_info[fold] = val_info

    def set_test_info(self):
        self.test_info = [None] * self.folds_num

        for fold in range(self.folds_num):
            fold_data_info_folder = os.path.join(self.data_info_folder, str(fold))

            test_info = pd.read_json(os.path.join(fold_data_info_folder, self.test_data_file_name))
            self.test_info[fold] = test_info

    def set_data_info(self):
        self.set_train_info()
        self.set_val_info()
        self.set_test_info()

    def _fill_fold_info(self, train_info, val_info, test_info, fold_idx=0, cls_idx=0):
        if cls_idx == 0:
            self.train_info[fold_idx] = train_info
            self.val_info[fold_idx] = val_info
            self.test_info[fold_idx] = test_info
        else:
            self.train_info[fold_idx] = self.train_info[fold_idx].append(train_info)
            self.val_info[fold_idx] = self.val_info[fold_idx].append(val_info)
            self.test_info[fold_idx] = self.test_info[fold_idx].append(test_info)

    def _split_train_val_info(self, train_val_info):
        train_val_info = train_val_info.reindex(np.random.permutation(train_val_info.index))

        train_split = int(self.train_split * train_val_info.shape[0])
        train_info = train_val_info.iloc[:train_split]
        val_info = train_val_info.iloc[train_split:]

        return train_info, val_info

    def _calculate_leave_out_folds_values(self, data_info):

        if self.leave_out:

            if self.leave_out_values is None:
                leave_out_unique_values = data_info[self.leave_out_param].unique()
                self.logger.log("Leave out values are: {0}".format(leave_out_unique_values))

                if self.folds_num > 1:
                    self.leave_out_folds_values = np.array_split(leave_out_unique_values, self.folds_num)
                else:
                    self.leave_out_folds_values = [np.random.choice(leave_out_unique_values, int(len(leave_out_unique_values) * self.test_split), replace=False)]

            else:
                self.leave_out_folds_values = self.leave_out_values

    def _fill_info(self, data_info, cls_idx=0):
        if self.folds_num > 1:

            if self.leave_out:
                for fold_idx in range(self.folds_num):
                    test_info = data_info.loc[data_info[self.leave_out_param].isin(self.leave_out_folds_values[fold_idx])]

                    train_val_info = data_info.loc[~data_info[self.leave_out_param].isin(self.leave_out_folds_values[fold_idx])]
                    train_info, val_info = self._split_train_val_info(train_val_info)

                    self._fill_fold_info(train_info, val_info, test_info, fold_idx, cls_idx=cls_idx)

            elif self.set_test_data_info:
                for fold_idx in range(self.folds_num):
                    fold_data_info_folder = os.path.join(self.data_info_folder, str(fold_idx))
                    test_info = pd.read_json(os.path.join(fold_data_info_folder, self.test_data_file_name))

                    train_val_info = data_info.loc[~data_info[self.set_test_data_param].isin(test_info[self.set_test_data_param])]
                    train_info, val_info = self._split_train_val_info(train_val_info)

                    self._fill_fold_info(train_info, val_info, test_info, fold_idx, cls_idx=cls_idx)

            else:
                kf = KFold(n_splits=self.folds_num)
                for fold_idx, (train_idxs, test_idxs) in enumerate(kf.split(data_info)):
                    test_info = data_info.iloc[test_idxs]

                    train_val_info = data_info.iloc[train_idxs]
                    train_info, val_info = self._split_train_val_info(train_val_info)

                    self._fill_fold_info(train_info, val_info, test_info, fold_idx, cls_idx=cls_idx)

        else:

            data_info = data_info.reindex(np.random.permutation(data_info.index))
            if self.leave_out:
                test_info = data_info.loc[data_info[self.leave_out_param].isin(self.leave_out_folds_values[0])]
                train_val_info = data_info.loc[~data_info[self.leave_out_param].isin(self.leave_out_folds_values[0])]
                train_info, val_info = self._split_train_val_info(train_val_info)

                self._fill_fold_info(train_info, val_info, test_info, fold_idx=0, cls_idx=cls_idx)

            elif self.set_test_data_info:
                fold_idx = 0
                fold_data_info_folder = os.path.join(self.data_info_folder, str(fold_idx))
                test_info = pd.read_json(os.path.join(fold_data_info_folder, self.test_data_file_name))

                train_val_info = data_info.loc[~data_info[self.set_test_data_param].
                                               isin(test_info[self.set_test_data_param])]
                train_info, val_info = self._split_train_val_info(train_val_info)

                self._fill_fold_info(train_info, val_info, test_info, fold_idx, cls_idx=cls_idx)

            else:
                test_split = int(self.test_split * data_info.shape[0])
                test_info = data_info.iloc[:test_split]
                train_val_info = data_info.iloc[test_split:]
                train_info, val_info = self._split_train_val_info(train_val_info)

                self._fill_fold_info(train_info, val_info, test_info, fold_idx=0, cls_idx=cls_idx)

    def split_data(self):

        if self.set_info:
            self.set_data_info()

        else:
            self.train_info = [None] * self.folds_num
            self.val_info = [None] * self.folds_num
            self.test_info = [None] * self.folds_num

            np.random.seed(self.data_random_seed)

            self._calculate_leave_out_folds_values(self.dataset.original_info)

            data_info = self.dataset.filtered_info
            data_info = data_info.reindex(np.random.permutation(data_info.index))

            self._fill_info(data_info)

    def save_split_data(self, folder, fold_num):

        self.train_info[fold_num].to_json(os.path.join(folder, self.train_data_file_name))
        self.val_info[fold_num].to_json(os.path.join(folder, self.val_data_file_name))
        self.test_info[fold_num].to_json(os.path.join(folder, self.test_data_file_name))

    def get_sequence(self, run_mode, fold=0):

        """
        This method creates instance of keras.utils Sequence class that can be passed to model fit method
        The Sequence object created for current fold and according to run mode (training or validation)
        :param run_mode: run mode (can be training or validation)
        :param fold: current fold
        :return: Sequence object
        """

        self.sequence_args["dataset"] = self.dataset
        if run_mode == RunMode.TRAINING:
            self.sequence_args["data_info"] = self.train_info[fold]
        else:
            self.sequence_args["data_info"] = self.val_info[fold]

        sequence = Sequence()
        sequence.parse_args(params=self.sequence_args)
        sequence.initialize()

        return sequence

    def get_pair(self, run_mode, preprocess, augment, get_label=True, get_data=True, fold=0):

        if run_mode == RunMode.TRAINING:
            data_info = self.train_info[fold]
        elif run_mode == RunMode.VALIDATION:
            data_info = self.val_info[fold]
        elif run_mode == RunMode.TEST:
            data_info = self.test_info[fold]
        elif run_mode == RunMode.INFERENCE:
            data_info = self.dataset.inference_info[fold]
        else:
            raise ValueError("Unknown run mode: {0}".format(run_mode))

        inputs = list()
        labels = list()

        for idx, (_, info_row) in enumerate(data_info.iterrows()):

            data, label = self.dataset.get_pair(info_row, preprocess=preprocess, augment=augment, get_data=True, get_label=get_label, run_mode=run_mode)

            # Dataset 'get_pair' method can return several samples in the form of list (and not one as in usual case)
            # For example, it can happens when we crop patches from the single image during inference
            if isinstance(data, list):
                inputs = inputs + data
            else:
                inputs.append(data)

            labels.append(label)

        # If preprocess is True then we can assume that all inputs and labels are of the same shape and can be copied to numpy array
        if preprocess:
            return np.copy(inputs), np.copy(labels)
        else:
            return inputs, labels
