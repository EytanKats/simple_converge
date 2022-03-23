from utils.dataset_utils import subsample, oversample, balance
from datasets.DataframeImageCategoricalDataset import DataframeImageCategoricalDataset


default_settings = {
    'subsample': False,
    'oversample': False,
    'balance': False,
    'balancing_num': 5,
    'sampling_column': '',
    'sample_with_replacement': True,
    'image_path_column': 'path',
    'label_name_column': 'label',
    'labels': []
}


class BalancedDataframeImageCategoricalDataset(DataframeImageCategoricalDataset):

    """
    Define interface for dataset classes.
    Dataset classes will inherit from this base class and will be responsible for:
    - loading data and corresponding labels.
    - balancing data.
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

        super(BalancedDataframeImageCategoricalDataset, self).__init__(settings, dataframe, transform)

        self.original_dataframe = dataframe
        self.dataframe = None

        # Sample dataset
        self.sample_data()

    def sample_data(self):

        """
        Sample the dataset dataframe according to chosen strategy.
        :return: None.
        """

        if self.settings['oversample']:
            self.dataframe = oversample(self.original_dataframe, self.settings['sampling_column'])
        elif self.settings['subsample']:
            self.dataframe = subsample(self.original_dataframe, self.settings['sampling_column'])
        elif self.settings['balance']:
            self.dataframe = balance(self.original_dataframe, self.settings['sampling_column'],
                                     self.settings['balancing_num'], self.settings['sample_with_replacement'])
        else:
            self.dataframe = self.original_dataframe

    def save_dataframe(self, output_prefix):
        self.original_dataframe.to_csv(output_prefix + '.csv', index=False)

    def on_epoch_end(self):
        self.sample_data()
