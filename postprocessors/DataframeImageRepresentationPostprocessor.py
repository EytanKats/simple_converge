import os
import numpy as np
import pandas as pd
from loguru import logger
from scipy import special

from simple_converge.postprocessors.BasePostProcessor import BasePostProcessor


default_postprocessor_settings = {
    'activation': 'softmax',
    'embedding_labels_column': ''
}


class DataframeImageRepresentationPostprocessor(BasePostProcessor):

    def __init__(
            self,
            settings,
            mlops_task,
            dataframe):

        super(DataframeImageRepresentationPostprocessor, self).__init__(settings, mlops_task)

        self.dataset_df = dataframe
        self.features_list = list()

        self.features = None

    def postprocess_predictions(
            self,
            predictions,
            data,
            run_mode
    ):

        if self.settings['activation'] == 'sigmoid':
            predictions = special.expit(predictions)
        elif self.settings['activation'] == 'softmax':
            predictions = special.softmax(predictions, axis=-1)

        self.features = predictions
        self.features_list.append(predictions)

    def calculate_metrics(
            self,
            output_folder,
            fold,
            task
    ):

        pass

    def save_predictions(
            self,
            output_folder,
            fold,
            task,
            run_mode
    ):
        pass

    def calculate_total_metrics(
            self,
            output_folder,
            fold,
            task
    ):
        features = np.concatenate(self.features_list)


