import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from simple_converge.datasets.DataframeSupervisedDataset import DataframeSupervisedDataset


default_settings = {
    'image_path_column': 'path',
    'label_name_column': 'label',
    'labels': [],
    'get_image_as_numpy_array': False
}


class DataframeImageCategoricalDataset(DataframeSupervisedDataset, Dataset):

    def __init__(
            self,
            settings,
            dataframe,
            transform
    ):

        super(DataframeImageCategoricalDataset, self).__init__(
            settings,
            dataframe,
            transform
        )

    def get_data(self, df_row):
        with open(df_row[self.settings['image_path_column']], "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

            if self.settings['get_image_as_numpy_array']:
                img = np.asarray(img).astype(np.float32)

            return img

    def get_label(self, df_row):
        idx = np.where(np.array(self.settings['labels']) == df_row[self.settings['label_name_column']])[0][0]
        return idx
