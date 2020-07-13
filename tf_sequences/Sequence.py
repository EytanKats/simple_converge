import math
import numpy as np
import tensorflow as tf

from base.BaseObject import BaseObject


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
        self.dataset = None
        self.data_info = None

        self.batch_size = 4
        self.apply_augmentations = False

        self.multi_input = False
        self.multi_output = False
        self.inputs_num = 1
        self.outputs_num = 1

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Sequence, self).parse_args(**kwargs)

        if "dataset" in self.params.keys():
            self.dataset = self.params["dataset"]

        if "data_info" in self.params.keys():
            self.data_info = self.params["data_info"]

        if "batch_size" in self.params.keys():
            self.batch_size = self.params["batch_size"]

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

    def initialize(self):

        """
        This method does obligatory initialization of the object to behave as expected.
        Have to be called before first use
        :return: None
        """

        self.data_info = self.data_info.reindex(np.random.permutation(self.data_info.index))

    def __len__(self):

        """
        This method return the number of batches in the data sequence
        :return: number of batches
        """

        return int(math.ceil(self.data_info.shape[0]) / self.batch_size)

    def __getitem__(self, idx):

        """
        This method constructs the current batch of inputs and labels for model fitting
        :param idx: The index of the current batch
        :return: batch of inputs and labels
        """

        # Get fold_train_data slice for the current batch
        batch_data_info = self.data_info.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, :]

        inputs = []
        labels = []
        for _, info_row in batch_data_info.iterrows():

            augment = self.apply_augmentations
            data, label = self.dataset.get_pair(info_row, preprocess=True, augment=augment)

            inputs.append(data)
            labels.append(label)

        # Rearrange multi-input data
        if self.multi_input:
            inputs = self._rearrange_multi_io(inputs, self.inputs_num)
        else:
            inputs = np.array(inputs)

        # Rearrange multi-output data
        if self.multi_output:
            labels = self._rearrange_multi_io(labels, self.outputs_num)
        else:
            labels = np.array(labels)

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

        self.data_info = self.data_info.reindex(np.random.permutation(self.data_info.index))

    @staticmethod
    def _rearrange_multi_io(data, io_num):

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
