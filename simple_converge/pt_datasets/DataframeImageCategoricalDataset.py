from torch.utils.data import Dataset

from utils.dataset_utils import subsample, oversample, balance
from datasets.DataframeImageCategoricalDataset import DataframeImageCategoricalDataset as BaseDataframeImageCategoricalDataset


default_settings = {
    'image_path_column': 'path',
    'label_name_column': 'label',
    'labels': []
}


class DataframeImageCategoricalDataset(BaseDataframeImageCategoricalDataset, Dataset):

    """
    Dataset class will be responsible for:
    - loading data and corresponding labels.
    - transformation of data and corresponding labels.
    """
