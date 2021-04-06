import math
import numpy as np
import pandas as pd
import tensorflow as tf

from simple_converge.base.BaseObject import BaseObject


class Sequence(tf.keras.utils.Sequence, BaseObject):

    """
    This class can be passed as a training data to the fit method
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(Sequence, self).__init__()

        # Fields to be filled by parsing
        self.batch_size = 4
        self.steps_per_epoch = None
        self.apply_augmentations = False

        self.multi_input = False
        self.multi_output = False
        self.inputs_num = 1
        self.outputs_num = 1

        self.subsample = False
        self.oversample = False
        self.subsampling_column = ""
        self.oversampling_column = ""

        # Fields to be filled during execution
        self.dataset = None
        self.dataset_df = None

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Sequence, self).parse_args(**kwargs)

        if "batch_size" in self.params.keys():
            self.batch_size = self.params["batch_size"]

        if "steps_per_epoch" in self.params.keys():
            self.steps_per_epoch = self.params["steps_per_epoch"]

        if "apply_augmentations" in self.params.keys():
            self.apply_augmentations = self.params["apply_augmentations"]

        if "multi_input" in self.params.keys():
            self.multi_input = self.params["multi_input"]

        if "multi_output" in self.params.keys():
            self.multi_output = self.params["multi_output"]

        if "inputs_num" in self.params.keys():
            self.inputs_num = self.params["inputs_num"]

        if "outputs_num" in self.params.keys():
            self.outputs_num = self.params["outputs_num"]

        if "subsample" in self.params.keys():
            self.subsample = self.params["subsample"]

        if "oversample" in self.params.keys():
            self.oversample = self.params["oversample"]

        if "subsampling_column" in self.params.keys():
            self.subsampling_column = self.params["subsampling_column"]

        if "oversampling_column" in self.params.keys():
            self.oversampling_column = self.params["oversampling_column"]

    def initialize(self):

        """
        This method does obligatory initialization of the object to behave as expected.
        Have to be called before first use
        :return: None
        """

        if self.oversample:
            self.dataset_df = self.oversample_data(self.dataset_df)
        elif self.subsample:
            self.dataset_df = self.subsample_data(self.dataset_df)
        else:
            self.dataset_df = self.dataset_df.reindex(np.random.permutation(self.dataset_df.index))

    def set_dataset_df(self, dataset_df):

        """
        This method set dataset dataframe that will be used by the sequence
        :param dataset_df: dataset dataframe
        :return: None
        """

        self.dataset_df = dataset_df

    def set_dataset(self, dataset):

        """
        This method set dataset object that will be used by the sequence
        :param dataset: dataset object
        :return: None
        """

        self.dataset = dataset

    def __len__(self):

        """
        This method return the number of batches in the data sequence
        :return: number of batches
        """

        if self.steps_per_epoch is not None:
            return self.steps_per_epoch

        else:
            return int(math.ceil(self.dataset_df.shape[0]) / self.batch_size)

    def __getitem__(self, idx):

        """
        This method constructs the current batch of inputs and labels for model fitting
        :param idx: The index of the current batch
        :return: batch of inputs and labels
        """

        # Get fold_train_data slice for the current batch
        batch_dataset_df = self.dataset_df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, :]

        inputs = []
        labels = []
        for _, info_row in batch_dataset_df.iterrows():

            augment = self.apply_augmentations
            data, label = self.dataset.get_data_sample(info_row,
                                                       get_data=True,
                                                       get_label=True,
                                                       augment=augment,
                                                       preprocess=True)

            inputs.append(data)
            labels.append(label)

        # Rearrange multi-input data
        if self.multi_input:
            inputs = self.rearrange_multi_io(inputs, self.inputs_num)
        else:
            inputs = np.array(inputs)

        # Rearrange multi-output data
        if self.multi_output:
            labels = self.rearrange_multi_io(labels, self.outputs_num)
        else:
            labels = np.array(labels)

        # Permute the data on the end of the epoch
        if idx + 1 == len(self):
            self.dataset_df = self.dataset_df.reindex(np.random.permutation(self.dataset_df.index))

        return inputs, labels

    def __iter__(self):

        """
        This method defines iterator on the batches of the data sequence
        :return: batch of inputs and labels
        """

        for item in (self[i] for i in range(len(self))):
            yield item

    def on_epoch_end(self):

        """
        This method permutes the dataset on the end of each epoch
        :return: None
        """

        if self.oversample:
            dataset_df = self.oversample_data(self.dataset_df)
        elif self.subsample:
            dataset_df = self.subsample_data(self.dataset_df)
        else:
            dataset_df = self.dataset_df

        self.dataset_df = dataset_df.reindex(np.random.permutation(dataset_df.index))

    @staticmethod
    def rearrange_multi_io(data, io_num):

        """
        This method rearrange data to match the tf.keras backend multi-input/multi-output format
        The format is changed from [ [io_11, io_21, ...], [io_12, io_22, ...], ... ] to [ [io_11, io_12, ...], [io_21, io_22, ...], ... ]
        :param data: multi-input/multi-output data
        :param io_num: number of inputs/outputs
        :return: rearranged multi-input/multi-output data
        """

        # Create placeholder for multi input/output
        rearranged_data = list()
        for _ in range(io_num):
            rearranged_data.append(list())

        for batch_item in data:  # iterate on data items in batch

            for io_idx in range(io_num):  # iterate on inputs/outputs in batch item
                rearranged_data[io_idx].append(batch_item[io_idx])

        rearranged_data = [np.array(data_list) for data_list in rearranged_data]
        return rearranged_data

    def subsample_data(self, dataset_df):

        # Find number of samples in the smallest class
        unique_values = dataset_df[self.subsampling_column].unique()
        rows_cnt = []
        for value in unique_values:
            value_info = dataset_df.loc[dataset_df[self.subsampling_column] == value]
            rows_cnt.append(value_info.shape[0])
        min_count = np.min(rows_cnt)

        # Subsample equal number of samples from all the classes
        subsampled_dataset_df = pd.DataFrame()
        for value in unique_values:
            value_info = dataset_df.loc[dataset_df[self.subsampling_column] == value]
            value_info_subsampled = value_info.sample(n=min_count)
            subsampled_dataset_df = subsampled_dataset_df.append(value_info_subsampled)

        return subsampled_dataset_df

    def oversample_data(self, dataset_df):

        # Find number of samples in the largest class
        unique_values = dataset_df[self.oversampling_column].unique()
        rows_cnt = []
        for value in unique_values:
            value_info = dataset_df.loc[dataset_df[self.oversampling_column] == value]
            rows_cnt.append(value_info.shape[0])
        max_count = np.max(rows_cnt)

        # Oversample equal number of samples from all the classes
        oversampled_dataset_df = pd.DataFrame()
        for value in unique_values:
            value_info = dataset_df.loc[dataset_df[self.oversampling_column] == value]
            value_info_oversampled = value_info.sample(n=max_count, replace=True)
            oversampled_dataset_df = oversampled_dataset_df.append(value_info_oversampled)

        oversampled_dataset_df = oversampled_dataset_df.reset_index(drop=True)  # prevent duplicate indexes

        return oversampled_dataset_df
