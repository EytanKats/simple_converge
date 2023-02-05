import os
import shutil
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from loguru import logger
from scipy import special
from sklearn.manifold import TSNE

from simple_converge.postprocessors.BasePostProcessor import BasePostProcessor


default_postprocessor_settings = {
    'activation': '',
    'num_of_closest_samples': '',
    'image_path_column': '',
    'tsne_columns': []
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

        # Get indices of images close in features space according to cosine distance
        logger.info('Calculate cosine distances between samples')
        norm_features = features / np.linalg.norm(features, axis=1).reshape((features.shape[0], 1)).repeat(features.shape[1], axis=1)
        distances = np.einsum('nc,mc->nm', norm_features, norm_features)
        closest_indices = np.fliplr(np.argsort(distances, axis=1)[:, -(self.settings['num_of_closest_samples'] + 1):-1])

        # Save close samples
        logger.info('Save close samples')
        for query_idx in range(closest_indices.shape[0]):

            output_image_folder = pathlib.Path(output_folder).joinpath(self._get_name_of_dir_for_closest_images(query_idx))
            os.makedirs(output_image_folder)

            output_query_image_path = output_image_folder.joinpath(self._get_closest_image_name(query_idx, 0))
            shutil.copy(self.dataset_df.iloc[query_idx][self.settings['image_path_column']], output_query_image_path)

            for similarity_idx, key_idx in enumerate(closest_indices[query_idx, :]):
                output_key_image_path = output_image_folder.joinpath(self._get_closest_image_name(key_idx, similarity_idx + 1))
                shutil.copy(self.dataset_df.iloc[key_idx][self.settings['image_path_column']], output_key_image_path)

        # Plot 2D TSNE for features
        logger.info('Plot 2D TSNE')
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(features)
        self.dataset_df['tsne_2d_one'] = tsne_results[:, 0]
        self.dataset_df['tsne_2d_two'] = tsne_results[:, 1]

        for column in self.settings['tsne_columns']:
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x='tsne_2d_one',
                y='tsne_2d_two',
                hue=column,
                palette=sns.color_palette('hls', self.dataset_df[column].unique().shape[0]),
                data=self.dataset_df,
                legend='full',
                alpha=1
            )

            output_path = pathlib.Path(output_folder).joinpath(f'tsne_{column}.png')
            plt.savefig(output_path)

        # Save updated data file with TSNE results
        logger.info('Save updated data file')
        self.dataset_df.to_csv(pathlib.Path(output_folder).joinpath('test_data_output.csv'), index=False)

    @staticmethod
    def _get_name_of_dir_for_closest_images(idx):
        return f'{idx}'

    @staticmethod
    def _get_closest_image_name(idx, rating):
        return f'r{rating}_{idx}.png'


