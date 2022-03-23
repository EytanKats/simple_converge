import math
import numpy as np
import tensorflow as tf


default_settings = {
    'batch_size': 16,
    'permute_data': False,
    'multi_input': False,
    'multi_output': False,
    'inputs_num': 1,
    'outputs_num': 1
}


class DataLoader(tf.keras.utils.Sequence):

    """
    Define data sequence for training loop.
    Class is responsible for:
    - batching the data for model training.
    """

    def __init__(self, settings, dataset):

        """
        This method initializes parameters.
        :param settings: dictionary that contains dataset settings.
        :param dataset: instance of dataset class that implements __get_item__() and __len__() methods.
        :return: None.
        """

        super(DataLoader, self).__init__()

        self.settings = settings

        # Fields to be filled during execution
        self.dataset = dataset
        self.indices = np.arange(len(self.dataset))

        # Permute indices of dataset
        self.permute()

    def __len__(self):

        """
        Return the number of batches in the data sequence.
        :return: number of batches.
        """

        return int(math.ceil(len(self.dataset) / self.settings['batch_size']))

    def __getitem__(self, idx):

        """
        Construct the current batch of inputs and labels for model training.
        :param idx: The index of the current batch.
        :return: Batch of inputs and labels.
        """

        inputs = []
        labels = []
        metadata = []
        for data_idx in self.indices[idx * self.settings['batch_size']:(idx + 1) * self.settings['batch_size']]:

            data, label, metadata_ = self.dataset[data_idx]
            inputs.append(data)
            labels.append(label)
            metadata.append(metadata_)

        # Rearrange multi-input data
        if self.settings['multi_input']:
            inputs = self.rearrange_multi_io(inputs, self.settings['inputs_num'])
        else:
            inputs = np.array(inputs)

        # Rearrange multi-output data
        if self.settings['multi_output']:
            labels = self.rearrange_multi_io(labels, self.settings['outputs_num'])
        else:
            labels = np.array(labels)

        # Permute the data on the end of the epoch
        if idx + 1 == len(self):
            self.permute()
            self.dataset.on_epoch_end()

        return inputs, labels, metadata

    def __call__(self):

        """
        This method defines 'call' method of the sequence to get batch of model inputs and corresponding labels
        :return: Batch of inputs and labels
        """

        for item in (self[i] for i in range(len(self))):
            yield item

    def permute(self):

        """
        Permute indices of the dataset.
        :return: None.
        """

        if self.settings['permute_data']:
            self.indices = np.random.permutation(len(self.dataset))

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
