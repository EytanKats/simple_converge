import abc

from simple_converge.utils.RunMode import RunMode
from simple_converge.base.BaseObject import BaseObject


class BaseDataset(BaseObject):

    """
    This class defines common methods for datasets
    and responsible for data manipulations:
    - loading data and ground truth labels
    - augmentation
    - preprocessing
    - postprocessing
    - calculating metrics
    - saving predictions
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(BaseDataset, self).__init__()

    def parse_args(self,
                   **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseDataset, self).parse_args(**kwargs)

    @abc.abstractmethod
    def get_data(self,
                 df_row,
                 run_mode=RunMode.TRAINING):

        """
        This method loads data
        :param df_row: row in the dataset dataframe that describes data sample
        :param run_mode: enumeration that specifies execution mode - training, validation, test or inference
        :return: single data sample
        """

        pass

    @abc.abstractmethod
    def get_label(self,
                  df_row,
                  run_mode=RunMode.TRAINING):

        """
        This method loads ground truth label
        :param df_row: row in the dataset dataframe that describes data sample
        :param run_mode: enumeration that specifies execution mode - training, validation, test or inference
        :return: single ground truth label
        """

        pass

    @abc.abstractmethod
    def apply_augmentations(self,
                            data,
                            label=None,
                            info_row=None,
                            run_mode=RunMode.TRAINING):

        """
        This method applies augmentations on data sample and corresponding ground truth label
        :param data: single data sample
        :param label: single ground truth label
        :param info_row: row in the dataset dataframe that describes data sample
        :param run_mode: enumeration that specifies execution mode - training, validation, test or inference
        :return: augmented data sample and corresponding augmented ground truth label
        """

        pass

    @abc.abstractmethod
    def apply_preprocessing(self,
                            data,
                            label=None,
                            info_row=None,
                            run_mode=RunMode.TRAINING):

        """
        This method applies preprocessing on data sample and corresponding ground truth label
        :param data: single data sample
        :param label: single ground truth label
        :param info_row: row in the dataset dataframe that describes data sample
        :param run_mode: enumeration that specifies execution mode - training, validation, test or inference
        :return: preprocessed data sample and corresponding preprocessed ground truth label
        """

        pass

    def get_data_sample(self,
                        df_row,
                        get_data=True,
                        get_label=False,
                        augment=False,
                        preprocess=False,
                        run_mode=RunMode.TRAINING):

        """
        This method loads single data sample and corresponding ground truth label, augment and preprocess it
        :param df_row: row in the dataset dataframe that describes data sample
        :param get_data: boolean flag; if True data is loaded
        :param get_label: boolean flag; if True label is loaded
        :param augment: boolean flag; if True data and label are augmented
        :param preprocess: boolean flag; if True data and label are preprocessed
        :param run_mode: enumeration that specifies execution mode - training, validation, test or inference
        :return: data sample and corresponding ground truth label
        """

        data = None
        label = None

        if get_data:
            data = self.get_data(df_row, run_mode)

        if get_label:
            label = self.get_label(df_row, run_mode)

        if augment:
            data, label = self.apply_augmentations(data, label, df_row, run_mode)

        if preprocess:
            data, label = self.apply_preprocessing(data, label, df_row, run_mode)

        return data, label

    def get_data_batch(self,
                       batch_df,
                       get_data=True,
                       get_label=False,
                       augment=False,
                       preprocess=False,
                       run_mode=RunMode.TRAINING):

        """
        This method loads batch of data sample and corresponding ground truth labels, augment and preprocess them
        :param batch_df: dataframe that describes batch of data
        :param get_data: boolean flag; if True data is loaded
        :param get_label: boolean flag; if True labels is loaded
        :param augment: boolean flag; if True data and labels are augmented
        :param preprocess: boolean flag; if True data and labels are preprocessed
        :param run_mode: enumeration that specifies execution mode - training, validation, test or inference
        :return: list of data samples and list of corresponding ground truth labels
        """

        data = list()
        labels = list()

        for idx, (_, df_row) in enumerate(batch_df.iterrows()):

            data_sample, label = self.get_data_sample(df_row=df_row,
                                                      get_data=get_data,
                                                      get_label=get_label,
                                                      augment=augment,
                                                      preprocess=preprocess,
                                                      run_mode=run_mode)

            data.append(data_sample)
            labels.append(label)

        return data, labels

    @abc.abstractmethod
    def apply_postprocessing_on_predictions_batch(self,
                                                  predictions,
                                                  preprocessed_data_and_labels=None,
                                                  not_preprocessed_data_and_labels=None,
                                                  batch_df=None,
                                                  batch_id=0,
                                                  run_mode=RunMode.TEST):

        """
        This method applies postprocessing on model predictions
        :param predictions: model output
        :param preprocessed_data_and_labels: model input; tuple (data, labels)
        :param not_preprocessed_data_and_labels: input to model before preprocessing; tuple (data, labels)
        :param batch_df: dataframe that describes batch of data with 'data_id'
        :param batch_id: number that identifies batch of data
        :param run_mode: enumeration that specifies execution mode - test or inference
        :return: postprocessed model output
        """

        pass

    @abc.abstractmethod
    def calculate_batch_metrics(self,
                                postprocessed_predictions,
                                preprocessed_data_and_labels,
                                not_preprocessed_data_and_labels=None,
                                batch_df=None,
                                batch_id=0,
                                output_dir=None):

        """
        This method calculates metrics for batch of data
        :param postprocessed_predictions: postprocessed model output
        :param preprocessed_data_and_labels: model input; tuple (data, labels)
        :param not_preprocessed_data_and_labels: input to model before preprocessing; tuple (data, labels)
        :param batch_df: dataframe that describes batch of data with 'data_id'
        :param batch_id: number that identifies batch of data
        :param output_dir: directory to save plots / images
        :return None
        """

        pass

    @abc.abstractmethod
    def aggregate_metrics_for_all_batches(self,
                                          output_dir=None):

        """
        This method aggregates metrics for all batches of data to get results for all test data
        param output_dir: directory to save plots / images
        return: None
        """

        pass

    @abc.abstractmethod
    def save_data_batch(self,
                        postprocessed_predictions,
                        output_dir,
                        not_postprocessed_predictions=None,
                        preprocessed_data_and_labels=None,
                        not_preprocessed_data_and_labels=None,
                        batch_df=None,
                        batch_id=0):

        """
        This method calculates metrics for batch of data
        :param postprocessed_predictions: postprocessed model output
        :param output_dir: directory to save plots / images
        :param not_postprocessed_predictions: model output before postprocessing
        :param preprocessed_data_and_labels: model input; tuple (data, labels)
        :param not_preprocessed_data_and_labels: input to model before preprocessing; tuple (data, labels)
        :param batch_df: dataframe that describes batch of data with 'data_id'
        :param batch_id: number that identifies batch of data
        :return None
        """

        pass
