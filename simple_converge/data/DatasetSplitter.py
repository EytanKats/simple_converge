import os
import sys
import numpy as np
import pandas as pd

from simple_converge.utils import dataset_utils
from simple_converge.utils.RunMode import RunMode
from simple_converge.logs.LogLevels import LogLevels
from simple_converge.base.BaseObject import BaseObject


class DatasetSplitter(BaseObject):

    """
    This class responsible to split dataset to folds
     and farther split each fold to training, validation and test partitions.
    Features:
     - samples for each internal group in dataset are split in the same manner between training,
      validation and test partitions.
     - samples that are belong to fold leave-out will be presented only in test partition for this fold.
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(DatasetSplitter, self).__init__()

        # Fields to be filled by parsing
        self.data_definition_file_path = ""

        self.folds_num = 1
        self.data_random_seed = 1509
        self.train_val_fraction = 0.8
        self.train_fraction = 0.8

        self.split_to_groups = False
        self.group_column = ""
        self.group_ids = None

        self.leave_out = False
        self.leave_out_column = ""
        self.leave_out_values = None

        self.train_df_file_name = "train_df.csv"
        self.val_df_file_name = "val_df.csv"
        self.test_df_file_name = "test_df.csv"

        # Fields to be filled during execution
        self.dataset_df = None
        self.groups_df_list = None
        self.train_df_list = None
        self.val_df_list = None
        self.test_df_list = None

    def parse_args(self,
                   **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(DatasetSplitter, self).parse_args(**kwargs)

        if "data_definition_file_path" in self.params.keys():
            self.data_definition_file_path = self.params["data_definition_file_path"]

        if "folds_num" in self.params.keys():
            self.folds_num = self.params["folds_num"]

        if "data_random_seed" in self.params.keys():
            self.data_random_seed = self.params["data_random_seed"]

        if "train_val_fraction" in self.params.keys():
            self.train_val_fraction = self.params["train_val_fraction"]

        if "train_fraction" in self.params.keys():
            self.train_fraction = self.params["train_fraction"]

        if "split_to_groups" in self.params.keys():
            self.split_to_groups = self.params["split_to_groups"]

        if "group_column" in self.params.keys():
            self.group_column = self.params["group_column"]

        if "group_ids" in self.params.keys():
            self.group_ids = self.params["group_ids"]

        if "leave_out" in self.params.keys():
            self.leave_out = self.params["leave_out"]

        if "leave_out_column" in self.params.keys():
            self.leave_out_column = self.params["leave_out_column"]

        if "leave_out_values" in self.params.keys():
            self.leave_out_values = self.params["leave_out_values"]

        if "train_df_file_name" in self.params.keys():
            self.train_df_file_name = self.params["train_df_file_name"]

        if "val_df_file_name" in self.params.keys():
            self.val_df_file_name = self.params["val_df_file_name"]

        if "test_df_file_name" in self.params.keys():
            self.test_df_file_name = self.params["test_df_file_name"]

    def initialize(self,
                   run_mode=RunMode.TRAINING):

        """
        This method loads dataset file and initializes lists of training, validation and test dataframes
        :param run_mode: enumeration that specifies execution mode - training, validation, test or inference
        :return: None
        """

        if self.data_definition_file_path:
            self.logger.log("Loading dataset file {0}".format(self.data_definition_file_path))
            self.dataset_df = dataset_utils.load_dataset_file(self.data_definition_file_path)
            self.logger.log("Dataset contains {0} entries".format(self.dataset_df.shape[0]))
        else:
            self.logger.log("Data definition file path is not specified")

        if run_mode != run_mode.INFERENCE:
            self.train_df_list = [None] * self.folds_num
            self.val_df_list = [None] * self.folds_num
            self.test_df_list = [None] * self.folds_num

    def set_training_dataframe(self,
                               training_df,
                               fold_num):

        """
        This method sets training dataframe
        :param training_df: training dataframe
        :param fold_num: fold number to set training dataframe for
        :return: None
        """

        self.train_df_list[fold_num] = training_df
        self.logger.log("Training dataframe with {0} entries is set for fold {1}".format(training_df.shape[0], fold_num))

    def set_validation_dataframe(self,
                                 validation_df,
                                 fold_num):

        """
        This method sets training dataframe
        :param validation_df: training dataframe
        :param fold_num: fold number to set training dataframe for
        :return: None
        """

        self.val_df_list[fold_num] = validation_df
        self.logger.log("Validation dataframe with {0} entries is set for fold {1}".format(validation_df.shape[0], fold_num))

    def set_test_dataframe(self,
                           test_df,
                           fold_num):

        """
        This method sets training dataframe
        :param test_df: training dataframe
        :param fold_num: fold number to set training dataframe for
        :return: None
        """

        self.test_df_list[fold_num] = test_df
        self.logger.log("Test dataframe with {0} entries is set for fold {1}".format(test_df.shape[0], fold_num))

    def split_dataset(self):

        """
        This method first split dataset to folds
        and farther split each fold to training, validation and test partitions
        :return: None
        """

        # Set random seed to ensure reproducibility of dataset partitioning across experiments on same hardware
        np.random.seed(self.data_random_seed)

        # Split dataset to groups
        if self.split_to_groups:
            self.split_dataset_to_groups()
        else:
            self.groups_df_list = [self.dataset_df]

        # Permute entries in each group
        self.groups_df_list = [group_df.reindex(np.random.permutation(group_df.index)) for group_df in self.groups_df_list]

        # Split dataset to folds and training, validation and test partitions for each fold
        if self.leave_out:

            # Choose unique leave-out values for each fold
            if self.leave_out_values is None:
                self.choose_leave_out_values()

            # Split dataset to folds based on leave-out values
            self.split_dataset_to_folds_with_leave_out()

        else:

            # Split dataset to folds in random manner
            self.split_dataset_to_folds_randomly()

    def split_dataset_to_groups(self):

        """
        # This method splits dataset to groups based on values of 'self.group_column'.
        # Samples in each group are split in same manner between training, validation and test partitions.
        # This is important, for example, to ensure that each class (in classification problem) is represented
        # in training, validation and test partition.
        """

        self.logger.log("Dividing dataset to groups based on values of '{0}' dataset column".format(self.group_column))

        # Get groups identifiers
        if self.group_ids is None:
            group_ids = self.dataset_df[self.group_column].unique()
        else:
            group_ids = self.group_ids

        self.logger.log("Dataset groups are: {0}".format(group_ids))

        # Split dataset to groups
        self.groups_df_list = [self.dataset_df[self.dataset_df[self.group_column] == unique_group_id] for unique_group_id in group_ids]
        for group_idx, group_df in enumerate(self.groups_df_list):
            self.logger.log("Group {0} contains {1} samples".format(group_ids[group_idx], group_df.shape[0]))

    def choose_leave_out_values(self):

        """
        This method chooses leave-out values for each fold.
        Leave-out values calculated based on values of 'self.leave_out_column'.
        Dataset entries which 'self.leave_out_column' value is one of calculated leave-out values
        for specific fold will present only in test partition for this fold.
        :return: None
        """

        self.logger.log("Choosing leave-out values for each fold from unique values of '{0}' dataset column".format(self.leave_out_column))

        # Get unique values for dataset leave-out column
        unique_values = self.dataset_df[self.leave_out_column].unique()
        self.logger.log("Group values are: {0}".format(unique_values))

        # Check that number of unique leave-out values are greater or equal to number of folds
        if len(unique_values) < self.folds_num:
            self.logger.log("Number of unique leave-out values are smaller than number of required folds", level=LogLevels.ERROR)
            sys.exit(1)

        # Get list of unique leave-out values for each fold
        if self.folds_num > 1:
            self.leave_out_values = np.array_split(unique_values, self.folds_num)
        else:
            self.leave_out_values = [np.random.choice(unique_values, int(len(unique_values) * (1 - self.train_val_fraction)), replace=False)]

        for fold in range(0, self.folds_num):
            self.logger.log("Leave out values for fold {0} are: {1}".format(fold, self.leave_out_values[fold]))

    def split_dataset_to_folds_with_leave_out(self):

        """
        This method splits dataset to folds and training, validation and test partitions for each fold based on leave-out values.
        Samples in each group are split in same manner between training, validation and test partitions.
        Leave-out values will be presented only in test partition of corresponding fold.
        """

        self.logger.log("Split dataset to folds and training, validation and test partitions for each fold based on leave-out values")

        for fold in range(0, self.folds_num):

            groups_train_df_list = list()
            groups_val_df_list = list()
            groups_test_df_list = list()
            for group_idx, group_df in enumerate(self.groups_df_list):

                group_test_df = group_df[group_df[self.leave_out_column].isin(self.leave_out_values[fold])]
                if group_test_df.shape[0] == 0:
                    self.logger.log("Group {0} hasn't any of leave out values: {1}".format(group_idx, self.leave_out_values[fold]), level=LogLevels.WARNING)
                else:
                    groups_test_df_list.append(group_test_df)

                group_train_val_df = group_df[~group_df[self.leave_out_column].isin(self.leave_out_values[fold])]
                if group_train_val_df.shape[0] == 0:
                    self.logger.log("All samples of group {0} is in one of leave out values: {1}".format(group_idx, self.leave_out_values[fold]), level=LogLevels.WARNING)
                else:
                    train_split_idx = int(group_train_val_df.shape[0] * self.train_fraction)
                    groups_train_df_list.append(group_train_val_df.iloc[0:train_split_idx])
                    groups_val_df_list.append(group_train_val_df.iloc[train_split_idx:])

            self.train_df_list[fold] = pd.concat(groups_train_df_list)
            self.val_df_list[fold] = pd.concat(groups_val_df_list)
            self.test_df_list[fold] = pd.concat(groups_test_df_list)

            self.logger.log("For fold {0}:".format(fold))
            self.logger.log("  {0} training samples".format(self.train_df_list[fold].shape[0]))
            self.logger.log("  {0} validation samples".format(self.val_df_list[fold].shape[0]))
            self.logger.log("  {0} test samples".format(self.test_df_list[fold].shape[0]))

    def split_dataset_to_folds_randomly(self):

        """
        This method splits dataset to folds and training, validation and test partitions for each fold in random manner.
        Samples in each group are split in same manner between training, validation and test partitions.
        """

        self.logger.log("Split dataset to folds and training, validation and test partitions for each fold randomly")

        for fold in range(0, self.folds_num):

            groups_train_df_list = list()
            groups_val_df_list = list()
            groups_test_df_list = list()
            for group_idx, group_df in enumerate(self.groups_df_list):

                train_val_split_idx = int(group_df.shape[0] * self.train_val_fraction)
                group_train_val_df = group_df.iloc[0:train_val_split_idx]
                groups_test_df_list.append(group_df.iloc[train_val_split_idx:])

                train_split_idx = int(group_train_val_df.shape[0] * self.train_fraction)
                groups_train_df_list.append(group_train_val_df.iloc[0:train_split_idx])
                groups_val_df_list.append(group_train_val_df.iloc[train_split_idx:])

            self.train_df_list[fold] = pd.concat(groups_train_df_list)
            self.val_df_list[fold] = pd.concat(groups_val_df_list)
            self.test_df_list[fold] = pd.concat(groups_test_df_list)

            self.logger.log("For fold {0}:".format(fold))
            self.logger.log("  {0} training samples".format(self.train_df_list[fold].shape[0]))
            self.logger.log("  {0} validation samples".format(self.val_df_list[fold].shape[0]))
            self.logger.log("  {0} test samples".format(self.test_df_list[fold].shape[0]))

    def save_dataframes_for_fold(self, output_dir, fold):

        """
        This method saves training, validation and test dataframes for specific fold
        :param output_dir: directory to save dataframes
        :param fold: fold number
        """

        self.train_df_list[fold].to_csv(os.path.join(output_dir, self.train_df_file_name))
        self.val_df_list[fold].to_csv(os.path.join(output_dir, self.val_df_file_name))
        self.test_df_list[fold].to_csv(os.path.join(output_dir, self.test_df_file_name))
