import numpy as np
import tensorflow as tf

from simple_converge.logs.LogLevels import LogLevels
from simple_converge.base.BaseObject import BaseObject
from simple_converge.tf_sequences.Sequence import Sequence


class BaseModel(BaseObject):

    """
    This class defines common methods to all Tensorflow models
    """

    def __init__(self):
        
        """
        This method initializes parameters
        :return: None 
        """

        super(BaseModel, self).__init__()

        # Fields to be filled by parsing
        self.start_point_model = False
        self.start_point_model_path = ""
        self.compile_start_point_model = False

        self.regularizer_args = None
        self.loss_args = None
        self.optimizer_args = None
        self.metrics_args = None
        self.callbacks_args = None
        self.train_sequence_args = None
        self.val_sequence_args = None

        self.epochs = None
        self.val_steps = None

        self.prediction_batch_size = None

        # Fields to be filled during execution
        self.model = None
        self.train_sequence = None
        self.val_sequence = None

        self.metrics_collection = None
        self.callbacks_collection = None
        self.regularizers_collection = None
        self.optimizers_collection = None

    def parse_args(self, **kwargs):
        
        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseModel, self).parse_args(**kwargs)

        if "start_point_model" in self.params.keys():
            self.start_point_model = self.params["start_point_model"]

        if "start_point_model_path" in self.params.keys():
            self.start_point_model_path = self.params["start_point_model_path"]

        if "compile_start_point_model" in self.params.keys():
            self.compile_start_point_model = self.params["compile_start_point_model"]

        if "regularizer_args" in self.params.keys():
            self.regularizer_args = self.params["regularizer_args"]

        if "loss_args" in self.params.keys():
            self.loss_args = self.params["loss_args"]

        if "optimizer_args" in self.params.keys():
            self.optimizer_args = self.params["optimizer_args"]

        if "metrics_args" in self.params.keys():
            self.metrics_args = self.params["metrics_args"]

        if "callbacks_args" in self.params.keys():
            self.callbacks_args = self.params["callbacks_args"]

        if "train_sequence_args" in self.params.keys():
            self.train_sequence_args = self.params["train_sequence_args"]

        if "val_sequence_args" in self.params.keys():
            self.val_sequence_args = self.params["val_sequence_args"]

        if "epochs" in self.params.keys():
            self.epochs = self.params["epochs"]

        if "prediction_batch_size" in self.params.keys():
            self.prediction_batch_size = self.params["prediction_batch_size"]

    def set_metrics_collection(self, metrics_collection):

        """
        This method sets metrics collection that will be available for model training
        :param metrics_collection: dictionary of available metrics
        :return: None
        """

        self.metrics_collection = metrics_collection

    def set_callbacks_collection(self, callbacks_collection):

        """
        This method sets callbacks collection that will be available for model training
        :param callbacks_collection: dictionary of available callbacks
        :return: None
        """

        self.callbacks_collection = callbacks_collection

    def set_optimizers_collection(self, optimizers_collection):

        """
        This method sets optimizers collection that will be available for model training
        :param optimizers_collection: dictionary of available optimizers
        :return: None
        """

        self.optimizers_collection = optimizers_collection

    def set_regularizers_collection(self, regularizers_collection):

        """
        This method sets regularizers collection that will be available for model training
        :param regularizers_collection: dictionary of available regularizers
        :return: None
        """

        self.regularizers_collection = regularizers_collection

    def create_sequence(self, sequence_args, dataframe, dataset):

        """
        This method initializes data sequence that will be used in fit method
        :param sequence_args: dictionary that contains settings of sequence
        :param dataframe: dataframe that contains data information
        :param dataset: instance of 'Dataset' class
        :return: instance of 'Sequence' class
        """

        sequence = Sequence()
        sequence.parse_args(params=sequence_args)
        sequence.set_logger(self.logger)
        sequence.set_dataset_df(dataframe)
        sequence.set_dataset(dataset)

        sequence.initialize()
        return sequence

    def create_train_sequence(self, dataframe, dataset):

        """
        This method initializes training data sequence that will be used in fit method
        :param dataframe: dataframe that contains training data information
        :param dataset: instance of 'Dataset' class
        :return: None
        """

        self.train_sequence = self.create_sequence(self.train_sequence_args, dataframe, dataset)

    def create_val_sequence(self, dataframe, dataset):

        """
        This method initializes validation data sequence that will be used in fit method
        :param dataframe: dataframe that contains validation data information
        :param dataset: instance of 'Dataset' class
        :return: None
        """

        self.val_sequence = self.create_sequence(self.val_sequence_args, dataframe, dataset)

    def build(self):
        pass

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def save_model(self, model_path):
        self.model.save(model_path)

    def _get_regularizer(self):

        """
        This method retrieves regularizer using regularizer arguments
        :return: regularizer
        """

        if self.regularizer_args is None:
            return None

        regularizer_class = self.regularizers_collection[self.regularizer_args["regularizer_name"]]
        regularizer_obj = regularizer_class()
        regularizer_obj.parse_args(params=self.regularizer_args)
        regularizer_fn = regularizer_obj.get_regularizer()

        return regularizer_fn

    def _get_losses(self):

        """
        This method retrieves loss functions using losses arguments
        :return: list of loss functions
        """

        losses_fns = list()
        losses_weights = list()
        for loss_args in self.loss_args:
            loss_class = self.metrics_collection[loss_args["metric_name"]]
            loss_obj = loss_class()
            loss_obj.parse_args(params=loss_args)
            loss_fn = loss_obj.get_metric()
            losses_fns.append(loss_fn)
            losses_weights.append(loss_args["loss_weight"])

        return losses_fns, losses_weights

    def _get_optimizer(self):

        """
        This method retrieves optimizer using optimizer arguments
        :return: optimizer
        """

        optimizer_class = self.optimizers_collection[self.optimizer_args["optimizer_name"]]
        optimizer_obj = optimizer_class()
        optimizer_obj.parse_args(params=self.optimizer_args)
        optimizer_fn = optimizer_obj.get_optimizer()

        return optimizer_fn

    def _get_metrics(self):

        """
        This method retrieves metrics using metrics arguments
        :return: list of metrics
        """

        metrics_fns = list()
        for metrics_args_for_output in self.metrics_args:

            metrics_fns_for_output = list()
            for metric_args in metrics_args_for_output:
                metric_class = self.metrics_collection[metric_args["metric_name"]]
                metric_obj = metric_class()
                metric_obj.parse_args(params=metric_args)
                metric_fn = metric_obj.get_metric()
                metrics_fns_for_output.append(metric_fn)

            metrics_fns.append(metrics_fns_for_output)

        return metrics_fns

    def _get_callbacks(self):

        """
        This method retrieves callbacks using callbacks arguments
        :return: list of callbacks
        """

        callbacks_fns = list()
        for callback_args in self.callbacks_args:
            callback_class = self.callbacks_collection[callback_args["callback_name"]]
            callback_obj = callback_class()
            callback_obj.parse_args(params=callback_args)
            callback_fn = callback_obj.get_callback()
            callbacks_fns.append(callback_fn)

        return callbacks_fns

    def compile(self):

        losses_fns, losses_weights = self._get_losses()

        self.model.compile(loss=losses_fns,
                           loss_weights=losses_weights,
                           optimizer=self._get_optimizer(),
                           metrics=self._get_metrics())

    def fit(self):

        self.model.fit(
            x=self.train_sequence,
            epochs=self.epochs,
            callbacks=self._get_callbacks(),
            validation_data=self.val_sequence)

    def predict(self, data):

        predictions = self.model.predict(np.array(data), batch_size=self.prediction_batch_size, verbose=1)
        return predictions

    def evaluate(self, data, labels):

        results = self.model.evaluate(np.array(data), np.array(labels), batch_size=self.prediction_batch_size, verbose=1)
        return results

    def summary(self):
        self.logger.log("Model Architecture", level=LogLevels.DEBUG, console=True)
        self.model.summary(print_fn=lambda x: self.logger.log(x, level=LogLevels.DEBUG, console=True))
