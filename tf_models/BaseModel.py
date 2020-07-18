import abc
import tensorflow as tf

from base.BaseObject import BaseObject
from utils.RunMode import RunMode
from logger.LogLevels import LogLevels

from tf_regularizers.regularizers_collection import regularizers_collection
from tf_metrics.metrics_collection import metrics_collection
from tf_callbacks.callbacks_collection import callbacks_collection
from tf_optimizers.optimizers_collection import optimizers_collection


class BaseModel(BaseObject):

    """
    This abstract class defines common methods to all Tensorflow models
    """

    def __init__(self):
        
        """
        This method initializes parameters
        :return: None 
        """

        super(BaseModel, self).__init__()

        self.model = None

        self.generator = None

        self.load_weights_path = None
        self.save_weights_path = None

        self.inputs_signature = None
        self.outputs_signature = None
        self.saved_model_folder = None

        self.model_architecture_file_path = ""
        self.saved_model_folder_path = ""

        self.kernel_initializer = "he_normal"

        self.regularizer_args = None
        self.losses_args = None
        self.optimizer_args = None
        self.metrics_args = None

        self.steps_per_epoch = None
        self.epochs = None
        self.val_steps = None
        self.callbacks_args = None

        self.prediction_batch_size = None

    def parse_args(self, **kwargs):
        
        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseModel, self).parse_args(**kwargs)

        if "generator" in self.params.keys():
            self.generator = self.params["generator"]

        if "load_weights_path" in self.params.keys():
            self.load_weights_path = self.params["load_weights_path"]

        if "save_weights_path" in self.params.keys():
            self.save_weights_path = self.params["save_weights_path"]

        if "inputs_signature" in self.params.keys():
            self.inputs_signature = self.params["inputs_signature"]

        if "outputs_signature" in self.params.keys():
            self.outputs_signature = self.params["outputs_signature"]

        if "saved_model_folder" in self.params.keys():
            self.saved_model_folder = self.params["saved_model_folder"]

        if "model_architecture_file_path" in self.params.keys():
            self.model_architecture_file_path = self.params["model_architecture_file_path"]

        if "saved_model_folder_path" in self.params.keys():
            self.saved_model_folder_path = self.params["saved_model_folder_path"]

        if "kernel_initializer" in self.params.keys():
            self.kernel_initializer = self.params["kernel_initializer"]

        if "regularizer_args" in self.params.keys():
            self.regularizer_args = self.params["regularizer_args"]

        if "losses_args" in self.params.keys():
            self.losses_args = self.params["losses_args"]

        if "optimizer_args" in self.params.keys():
            self.optimizer_args = self.params["optimizer_args"]

        if "metrics_args" in self.params.keys():
            self.metrics_args = self.params["metrics_args"]

        if "steps_per_epoch" in self.params.keys():
            self.steps_per_epoch = self.params["steps_per_epoch"]

        if "epochs" in self.params.keys():
            self.epochs = self.params["epochs"]

        if "val_steps" in self.params.keys():
            self.val_steps = self.params["val_steps"]

        if "callbacks_args" in self.params.keys():
            self.callbacks_args = self.params["callbacks_args"]

        if "prediction_batch_size" in self.params.keys():
            self.prediction_batch_size = self.params["prediction_batch_size"]

    @abc.abstractmethod
    def build(self):
        pass

    def load_weights(self):
        self.model.load_weights(self.load_weights_path)

    def save_weights(self):
        self.model.save_weights(self.save_weights_path)

    def save_tf_model(self):
        signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={self.inputs_signature:
                                                                                     self.model.input},
                                                                             outputs={self.outputs_signature:
                                                                                      self.model.output})

        builder = tf.saved_model.builder.SavedModelBuilder(self.saved_model_folder)
        builder.add_meta_graph_and_variables(sess=tf.keras.backend.get_session(),
                                             tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                 signature})
        builder.save()

    def save_model(self):
        self.model.save(self.saved_model_folder_path)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.saved_model_folder_path)

    def to_json(self):
        model_json = self.model.to_json()
        with open(self.model_architecture_file_path, "w") as json_file:
            json_file.write(model_json)
            
    def from_json(self):
        json_file = open(self.model_architecture_file_path, "r")
        json_model = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(json_model)
        
        return model

    def _get_regularizer(self):

        """
        This method retrieves regularizer using regularizer arguments
        :return: regularizer
        """

        if self.regularizer_args is None:
            return None

        regularizer_class = regularizers_collection[self.regularizer_args["regularizer_name"]]
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
        for loss_args in self.losses_args:
            loss_class = metrics_collection[loss_args["metric_name"]]
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

        optimizer_class = optimizers_collection[self.optimizer_args["optimizer_name"]]
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
                metric_class = metrics_collection[metric_args["metric_name"]]
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
            callback_class = callbacks_collection[callback_args["callback_name"]]
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

    def fit(self, fold=0):

        self.model.fit(
            x=self.generator.get_sequence(run_mode=RunMode.TRAINING, fold=fold),
            epochs=self.epochs,
            callbacks=self._get_callbacks(),
            validation_data=self.generator.get_sequence(run_mode=RunMode.VALIDATION, fold=fold))

    def predict(self, run_mode, fold=0):

        inputs, labels = self.generator.get_pair(run_mode,
                                                 preprocess=True,
                                                 augment=False,
                                                 get_label=False,
                                                 get_data=True,
                                                 fold=fold)

        self.load_weights()
        predictions = self.model.predict(inputs, batch_size=self.prediction_batch_size, verbose=1)

        return predictions

    def evaluate(self, run_mode, fold=0):

        inputs, labels = self.generator.get_pair(run_mode,
                                                 preprocess=True,
                                                 augment=False,
                                                 get_label=True,
                                                 get_data=True,
                                                 fold=fold)

        self.load_weights()
        results = self.model.evaluate(inputs, labels, batch_size=self.prediction_batch_size, verbose=1)

        return results

    def summary(self):
        self._log("Model Architecture", level=LogLevels.DEBUG, console=True)
        self.model.summary(print_fn=lambda x: self._log(x, level=LogLevels.DEBUG, console=True))