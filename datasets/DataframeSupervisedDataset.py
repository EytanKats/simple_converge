import abc
from simple_converge.datasets.BaseDataframeDataset import BaseDataframeDataset


default_settings = {
}


class DataframeSupervisedDataset(BaseDataframeDataset):

    """
    Define interface for dataset classes.
    Dataset classes will inherit from this base class and will be responsible for:
    - loading data and corresponding labels.
    - transformation of data and corresponding labels.
    """

    @abc.abstractmethod
    def get_data(self, df_row):

        """
        Load data
        :param df_row: row in the dataframe that describes data sample
        :return: single data sample
        """

        pass

    @abc.abstractmethod
    def get_label(self, df_row):

        """
        Load ground truth label
        :param df_row: row in the dataframe that describes data sample
        :return: single label
        """

        pass

    def __getitem__(self, index):

        """
        Get data sample and corresponding ground truth label
        :param index: index of the row in the dataframe that describes data sample
        :return: data sample and corresponding ground truth label
        """

        data = self.get_data(self.dataframe.iloc[index])
        label = self.get_label(self.dataframe.iloc[index])
        data = self.transform(data, label)
        data = list(data)  # there is an assumption that self.transform returns tuple
        data.append(index)

        return data

    def __len__(self):

        return self.dataframe.shape[0]
