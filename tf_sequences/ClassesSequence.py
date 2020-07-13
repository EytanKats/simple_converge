import numpy as np
import pandas as pd

from tf_sequences.Sequence import Sequence


class ClassesSequence(Sequence):

    """
    This class expands the Sequence class for categorical classification tasks
    This class adds the ability to oversample, subsample the dataset for each epoch
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(Sequence, self).__init__()

        # Fields to be filled by parsing
        self.subsample = False
        self.oversample = False
        self.subsampling_param = ""
        self.oversampling_param = ""

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Sequence, self).parse_args(**kwargs)

        if "subsample" in self.params.keys():
            self.subsample = self.params["subsample"]

        if "oversample" in self.params.keys():
            self.oversample = self.params["oversample"]

        if "subsampling_param" in self.params.keys():
            self.subsampling_param = self.params["subsampling_param"]

        if "oversampling_param" in self.params.keys():
            self.oversampling_param = self.params["oversampling_param"]

    def initialize(self):
        """
        This method does obligatory initialization of the object to behave as expected
        Have to be called before first use
        :return: None
        """

        if self.oversample:
            self.data_info = self._oversample_data(self.data_info)
        elif self.subsample:
            self.data_info = self._subsample_data(self.data_info)

    def on_epoch_end(self):

        """
        This method permutes the dataset on the end of each epoch
        :return: None
        """

        data_info = None
        if self.oversample:
            data_info = self._oversample_data(self.data_info)
        elif self.subsample:
            data_info = self._subsample_data(self.data_info)

        self.data_info = data_info.reindex(np.random.permutation(data_info.index))

    def _subsample_data(self, data_info):

        # Find number of samples in the smallest class
        unique_values = data_info[self.subsampling_param].unique()
        rows_cnt = []
        for value in unique_values:
            value_info = data_info.loc[data_info[self.subsampling_param] == value]
            rows_cnt.append(value_info.shape[0])
        min_count = np.min(rows_cnt)

        # Subsample equal number of samples from all the classes
        subsampled_data_info = pd.DataFrame()
        for value in unique_values:
            value_info = data_info.loc[data_info[self.subsampling_param] == value]
            value_info_subsampled = value_info.sample(n=min_count)
            subsampled_data_info = subsampled_data_info.append(value_info_subsampled)

        return subsampled_data_info

    def _oversample_data(self, data_info):

        # Find number of samples in the largest class
        unique_values = data_info[self.oversampling_param].unique()
        rows_cnt = []
        for value in unique_values:
            value_info = data_info.loc[data_info[self.oversampling_param] == value]
            rows_cnt.append(value_info.shape[0])
        max_count = np.max(rows_cnt)

        # Oversample equal number of samples from all the classes
        oversampled_data_info = pd.DataFrame()
        for value in unique_values:
            value_info = data_info.loc[data_info[self.oversampling_param] == value]
            value_info_oversampled = value_info.sample(n=max_count, replace=True)
            oversampled_data_info = oversampled_data_info.append(value_info_oversampled)

        oversampled_data_info = oversampled_data_info.reset_index(drop=True)  # prevent duplicate indexes
        return oversampled_data_info
