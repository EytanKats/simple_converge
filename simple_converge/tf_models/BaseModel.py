import numpy as np
import tensorflow as tf

from simple_converge.base.BaseObject import BaseObject
from simple_converge.logs.LogLevels import LogLevels


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
        self.load_weights_path = ""
        self.save_weights_path = ""

        self.load_model_path = ""
        self.save_model_path = ""

        self.kernel_initializer = "he_normal"

        self.regularizer_args = None
        self.loss_args = None
        self.optimizer_args = None
        self.metrics_args = None

        self.epochs = None
        self.val_steps = None
        self.callbacks_args = None

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

        if "load_weights_path" in self.params.keys():
            self.load_weights_path = self.params["load_weights_path"]

        if "save_weights_path" in self.params.keys():
            self.save_weights_path = self.params["save_weights_path"]

        if "load_model_path" in self.params.keys():
            self.load_model_path = self.params["load_model_path"]

        if "save_model_path" in self.params.keys():
            self.save_model_path = self.params["save_model_path"]

        if "kernel_initializer" in self.params.keys():
            self.kernel_initializer = self.params["kernel_initializer"]

        if "regularizer_args" in self.params.keys():
            self.regularizer_args = self.params["regularizer_args"]

        if "loss_args" in self.params.keys():
            self.loss_args = self.params["loss_args"]

        if "optimizer_args" in self.params.keys():
            self.optimizer_args = self.params["optimizer_args"]

        if "metrics_args" in self.params.keys():
            self.metrics_args = self.params["metrics_args"]

        if "epochs" in self.params.keys():
            self.epochs = self.params["epochs"]

        if "callbacks_args" in self.params.keys():
            self.callbacks_args = self.params["callbacks_args"]

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

    def set_train_sequence(self, train_sequence):

        """
        This method sets training data sequence that will be used in fit method
        :param train_sequence: training data sequence
        :return: None
        """

        self.train_sequence = train_sequence

    def set_val_sequence(self, val_sequence):

        """
        This method sets validation data sequence that will be used in fit method
        :param val_sequence: validation data sequence
        :return: None
        """

        self.val_sequence = val_sequence

    def build(self):
        pass

    def load_weights(self):
        self.model.load_weights(self.load_weights_path)

    def save_weights(self):
        self.model.save_weights(self.save_weights_path)

    def save_model(self):
        self.model.save(self.save_model_path)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.load_model_path)

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
