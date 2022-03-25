import sys
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import model_selection

from utils import dataset_utils


default_settings = {
    'data_definition_file_path': 'dataset.csv',
    'folds_num': 5,
    'data_random_seed': 1509,
    'train_val_fraction': 0.8,
    'train_fraction': 0.8,
    'split_to_groups': False,
    'group_column': '',
    'group_ids': None,
    'leave_out': False,
    'leave_out_column': '',
    'leave_out_values': None
}


class DatasetSplitter:

    """
    This class responsible to split dataset to folds
     and farther split each fold to training, validation and test partitions.
    Features:
     - samples for each internal group in dataset are split in the same manner between training,
      validation and test partitions.
     - samples that belong to fold leave-out will be presented only in test partition for this fold.
    """

    def __init__(self, settings):

        """
        This method initializes parameters
        :return: None
        """

        self.settings = settings

        self.dataset_df = None
        self.groups_df_list = None
        self.train_df_list = None
        self.val_df_list = None
        self.test_df_list = None

    def load_dataset_file(self):

        """
        This method loads dataset file
        :return: None
        """

        if self.settings['data_definition_file_path']:
            logger.info("Loading dataset file {0}".format(self.settings['data_definition_file_path']))
            self.dataset_df = dataset_utils.load_dataset_file(self.settings['data_definition_file_path'])
            logger.info("Dataset contains {0} entries".format(self.dataset_df.shape[0]))
        else:
            logger.info("Data definition file path is not specified")

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
        logger.info("Training dataframe with {0} entries is set for fold {1}".format(training_df.shape[0], fold_num))

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
        logger.info("Validation dataframe with {0} entries is set for fold {1}".format(validation_df.shape[0], fold_num))

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
        logger.info("Test dataframe with {0} entries is set for fold {1}".format(test_df.shape[0], fold_num))

    def set_custom_data_split(self, train_data_files, val_data_files, test_data_files):

        """
        This method sets training, validation and test dataframe lists according to custom lists of
        training, validation and test files defined in the settings.
        :return: None
        """

        logger.info("Loading custom lists of training validation and test files")

        self.train_df_list = [dataset_utils.load_dataset_file(data_file) for data_file in train_data_files]
        self.val_df_list = [dataset_utils.load_dataset_file(data_file) for data_file in val_data_files]
        self.test_df_list = [dataset_utils.load_dataset_file(data_file) for data_file in test_data_files]

    def split_dataset(self):

        """
        This method first split dataset to folds
        and farther split each fold to training, validation and test partitions
        :return: None
        """

        # Create lists to hold dataset partitions
        self.train_df_list = [None] * self.settings['folds_num']
        self.val_df_list = [None] * self.settings['folds_num']
        self.test_df_list = [None] * self.settings['folds_num']

        # Set random seed to ensure reproducibility of dataset partitioning across experiments on same hardware
        np.random.seed(self.settings['data_random_seed'])

        # Split dataset to groups
        if self.settings['split_to_groups']:
            self.split_dataset_to_groups()
        else:
            self.groups_df_list = [self.dataset_df]

        # Permute entries in each group
        self.groups_df_list = [group_df.reindex(np.random.permutation(group_df.index)) for group_df in self.groups_df_list]

        # Split dataset to folds and training, validation and test partitions for each fold
        if self.settings['leave_out']:

            # Choose unique leave-out values for each fold
            if self.settings['leave_out_values'] is None:
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

        logger.info("Dividing dataset to groups based on values of '{0}' dataset column".format(self.settings['group_column']))

        # Get groups identifiers
        if self.settings['group_ids'] is None:
            group_ids = self.dataset_df[self.settings['group_column']].unique()
        else:
            group_ids = self.settings['group_ids']

        logger.info("Dataset groups are: {0}".format(group_ids))

        # Split dataset to groups
        self.groups_df_list = [self.dataset_df[self.dataset_df[self.settings['group_column']] == unique_group_id] for unique_group_id in group_ids]
        for group_idx, group_df in enumerate(self.groups_df_list):
            logger.info("Group {0} contains {1} samples".format(group_ids[group_idx], group_df.shape[0]))

    def choose_leave_out_values(self):

        """
        This method chooses leave-out values for each fold.
        Leave-out values calculated based on values of 'self.leave_out_column'.
        Dataset entries which 'self.leave_out_column' value is one of calculated leave-out values
        for specific fold will present only in test partition for this fold.
        :return: None
        """

        logger.info("Choosing leave-out values for each fold from unique values of '{0}' dataset column".format(self.settings['leave_out_column']))

        # Get unique values for dataset leave-out column
        unique_values = self.dataset_df[self.settings['leave_out_column']].unique()
        logger.info("Unique values for column {0} are: {1}".format(self.settings['leave_out_column'], unique_values))

        # Check that number of unique leave-out values are greater or equal to number of folds
        if len(unique_values) < self.settings['folds_num']:
            logger.error("Number of unique leave-out values are smaller than number of required folds")
            sys.exit(1)

        # Get list of unique leave-out values for each fold
        if self.settings['folds_num'] > 1:
            self.settings['leave_out_values'] = np.array_split(unique_values, self.settings['folds_num'])
        else:
            self.settings['leave_out_values'] = [np.random.choice(unique_values, int(len(unique_values) * (1 - self.settings['train_val_fraction'])), replace=False)]

        for fold in range(0, self.settings['folds_num']):
            logger.info("Leave out values for fold {0} are: {1}".format(fold, self.settings['leave_out_values'][fold]))

    def split_dataset_to_folds_with_leave_out(self):

        """
        This method splits dataset to folds and training, validation and test partitions for each fold based on leave-out values.
        Samples in each group are split in same manner between training, validation and test partitions.
        Leave-out values will be presented only in test partition of corresponding fold.
        """

        logger.info("Split dataset to folds and training, validation and test partitions for each fold based on leave-out values")

        for fold in range(0, self.settings['folds_num']):

            groups_train_df_list = list()
            groups_val_df_list = list()
            groups_test_df_list = list()
            for group_idx, group_df in enumerate(self.groups_df_list):

                group_test_df = group_df[group_df[self.settings['leave_out_column']].isin(self.settings['leave_out_values'][fold])]
                if group_test_df.shape[0] == 0:
                    logger.warning("Group {0} hasn't any of leave out values: {1}".format(group_idx, self.settings['leave_out_values'][fold]))
                else:
                    groups_test_df_list.append(group_test_df)

                group_train_val_df = group_df[~group_df[self.settings['leave_out_column']].isin(self.settings['leave_out_values'][fold])]
                if group_train_val_df.shape[0] == 0:
                    logger.warning("All samples of group {0} is in one of leave out values: {1}".format(group_idx, self.settings['leave_out_values'][fold]))
                else:
                    train_split_idx = int(group_train_val_df.shape[0] * self.settings['train_fraction'])
                    groups_train_df_list.append(group_train_val_df.iloc[0:train_split_idx])
                    groups_val_df_list.append(group_train_val_df.iloc[train_split_idx:])

            self.train_df_list[fold] = pd.concat(groups_train_df_list)
            self.val_df_list[fold] = pd.concat(groups_val_df_list)
            self.test_df_list[fold] = pd.concat(groups_test_df_list)

        # Print number of examples in training, validation and test for each fold
        self.print_data_split()

    def split_dataset_to_folds_randomly(self):

        """
        This method splits dataset to folds and training, validation and test partitions for each fold in random manner.
        Samples in each group are split in same manner between training, validation and test partitions.
        """

        logger.info("Split dataset to folds and training, validation and test partitions for each fold randomly")

        # For one fold regime data will be divided according to training-validation fraction and training fraction
        # defined in settings.
        # For multiple folds regime data will be divided with use of sklearn module and according to training
        # fraction defined in settings

        if self.settings['folds_num'] == 1:

            groups_train_df_list = list()
            groups_val_df_list = list()
            groups_test_df_list = list()
            for group_df in self.groups_df_list:
                train_val_split_idx = int(group_df.shape[0] * self.settings['train_val_fraction'])
                group_train_val_df = group_df.iloc[0:train_val_split_idx]
                groups_test_df_list.append(group_df.iloc[train_val_split_idx:])

                train_split_idx = int(group_train_val_df.shape[0] * self.settings['train_fraction'])
                groups_train_df_list.append(group_train_val_df.iloc[0:train_split_idx])
                groups_val_df_list.append(group_train_val_df.iloc[train_split_idx:])

            self.train_df_list[0] = pd.concat(groups_train_df_list)
            self.val_df_list[0] = pd.concat(groups_val_df_list)
            self.test_df_list[0] = pd.concat(groups_test_df_list)

        else:

            # Split each group to multiple folds
            kf_list = list()
            kf = model_selection.KFold(n_splits=self.settings['folds_num'], shuffle=True, random_state=self.settings['data_random_seed'])
            for group_df in self.groups_df_list:
                kf_list.append(kf.split(group_df))

            # Combine group splits to folds
            for fold in range(0, self.settings['folds_num']):

                fold_split = [next(kf_list[idx]) for idx in range(len(kf_list))]

                groups_train_df_list = list()
                groups_val_df_list = list()
                groups_test_df_list = list()
                for group_idx, group_df in enumerate(self.groups_df_list):
                    group_train_val_df = group_df.iloc[fold_split[group_idx][0]]
                    groups_test_df_list.append(group_df.iloc[fold_split[group_idx][1]])

                    train_split_idx = int(group_train_val_df.shape[0] * self.settings['train_fraction'])
                    groups_train_df_list.append(group_train_val_df.iloc[0:train_split_idx])
                    groups_val_df_list.append(group_train_val_df.iloc[train_split_idx:])

                self.train_df_list[fold] = pd.concat(groups_train_df_list)
                self.val_df_list[fold] = pd.concat(groups_val_df_list)
                self.test_df_list[fold] = pd.concat(groups_test_df_list)

        # Print number of examples in training, validation and test for each fold
        self.print_data_split()

    def print_data_split(self):

        """
        This method prints number of examples in training, validation and test for each fold
        :return: None
        """

        for fold in range(0, self.settings['folds_num']):
            logger.info("For fold {0}:".format(fold))
            logger.info("  {0} training samples".format(self.train_df_list[fold].shape[0]))
            logger.info("  {0} validation samples".format(self.val_df_list[fold].shape[0]))
            logger.info("  {0} test samples".format(self.test_df_list[fold].shape[0]))
