import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from simple_converge.datasets.DataframeUnsupervisedDataset import DataframeUnsupervisedDataset


default_settings = {
    'image_path_column': 'path',
    'get_image_as_numpy_array': False
}


class DataframeImageDataset(DataframeUnsupervisedDataset, Dataset):

    def __init__(
            self,
            settings,
            dataframe,
            transform
    ):

        super(DataframeImageDataset, self).__init__(
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
