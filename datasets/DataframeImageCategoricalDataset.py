import cv2
import numpy as np
from datasets.BaseDataframeDataset import BaseDataframeDataset


default_settings = {
    'image_path_column': 'path',
    'label_name_column': 'label',
    'labels': []
}


class DataframeImageCategoricalDataset(BaseDataframeDataset):

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
        image = cv2.imread(df_row[self.settings['image_path_column']], flags=cv2.IMREAD_COLOR)
        return image

    def get_label(self, df_row):
        label = np.zeros(shape=(len(self.settings['labels']),))
        idx = np.where(np.array(self.settings['labels']) == df_row[self.settings['label_name_column']])[0][0]
        label[idx] = 1
        return label
