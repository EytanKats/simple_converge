import abc


class BasePostProcessor(abc.ABC):

    """
    Define interface for dataset classes.
    Postprocessor classes will inherit from this base class and will be responsible for:
    - postprocessing predictions
    - calculating metrics
    - saving predictions
    """

    def __init__(self, settings):

        """
        Initialize parameters.
        :param settings: dictionary that contains dataset settings.
        :return: None.
        """

        self.settings = settings

    @abc.abstractmethod
    def postprocess_predictions(
            self,
            predictions,
            data,
            run_mode
    ):

        """
        This method applies postprocessing on model predictions
        :param predictions: model output
        :param data: model input
        :param run_mode: enumeration that specifies execution mode - test or inference
        :return: postprocessed model output
        """

        pass

    @abc.abstractmethod
    def calculate_metrics(
            self,
            output_folder,
            fold,
            task
    ):

        """
        This method calculates metrics for batch of data
        :param output_folder: directory to save plots / images
        :param fold: number of fold
        :param task: instance of ClearML task
        :return None
        """

        pass

    @abc.abstractmethod
    def save_predictions(
            self,
            output_folder,
            fold,
            task,
            run_mode):

        """
        This method calculates metrics for batch of data
        :param output_folder: directory to save plots / images
        :param fold: number of fold
        :param task: instance of ClearML task
        :param run_mode: enumeration that specifies execution mode - test or inference
        :return None
        """

        pass

    @abc.abstractmethod
    def calculate_total_metrics(
            self,
            output_folder,
            fold,
            task):

        """
        This method aggregates metrics for all batches of data to get results for all fold data
        :param output_folder: directory to save plots / images
        :param fold: number of fold
        :param task: instance of ClearML task
        :return: None
        """

        pass

