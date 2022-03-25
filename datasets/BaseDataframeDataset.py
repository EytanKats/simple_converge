import abc
from utils.dataset_utils import subsample, oversample, balance


default_settings = {
}


class BaseDataframeDataset(abc.ABC):

    """
    Define interface for dataset classes.
    Dataset classes will inherit from this base class and will be responsible for:
    - loading data and corresponding labels.
    - transformation of data and corresponding labels.
    """

    def __init__(self, settings, dataframe, transform):

        """
        Initialize parameters.
        :param settings: dictionary that contains dataset settings.
        :param dataframe: dataframe each row of which describes data sample.
        :param transform: method that implements transform of single data sample and corresponding ground truth label.
        :return: None.
        """

        super(BaseDataframeDataset, self).__init__()

        self.settings = settings

        self.dataframe = dataframe
        self.transform = transform

    def save_dataframe(self, output_prefix):
        self.dataframe.to_csv(output_prefix + '.csv', index=False)

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
        data, label = self.transform(data, label)

        # return data, label, self.dataframe.iloc[index]
        return data, label, index

    def __len__(self):

        return self.dataframe.shape[0]
