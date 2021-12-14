import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm

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
        self.regularizer_args = None
        self.loss_args = None
        self.optimizer_args = None
        self.metrics_args = None
        self.train_sequence_args = None
        self.val_sequence_args = None

        self.epochs = None
        self.prediction_batch_size = None

        self.monitor = "not_defined"
        self.monitor_regime = "not_defined"

        self.ckpt_freq = 1
        self.ckpt_save_best_only = True
        self.ckpt_path = ""

        self.use_early_stopping = False
        self.early_stopping_patience = 10

        self.use_reduce_lr_on_plateau = False
        self.reduce_lr_on_plateau_patience = 3
        self.reduce_lr_on_plateau_factor = 0.8
        self.reduce_lr_on_plateau_min = 1e-6

        # Fields to be filled during execution
        self.model = None
        self.train_sequence = None
        self.val_sequence = None

        self.ckpt = None
        self.monitor_cur_val = 0
        self.monitor_best_val = 0
        self.ckpt_best_epoch = 0
        self.early_stopping_cnt = 0
        self.reduce_lr_on_plateau_cnt = 0

        self.metrics_collection = None
        self.regularizers_collection = None
        self.optimizers_collection = None

    def parse_args(self, **kwargs):
        
        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseModel, self).parse_args(**kwargs)

        if "regularizer_args" in self.params.keys():
            self.regularizer_args = self.params["regularizer_args"]

        if "loss_args" in self.params.keys():
            self.loss_args = self.params["loss_args"]

        if "optimizer_args" in self.params.keys():
            self.optimizer_args = self.params["optimizer_args"]

        if "metrics_args" in self.params.keys():
            self.metrics_args = self.params["metrics_args"]

        if "train_sequence_args" in self.params.keys():
            self.train_sequence_args = self.params["train_sequence_args"]

        if "val_sequence_args" in self.params.keys():
            self.val_sequence_args = self.params["val_sequence_args"]

        if "epochs" in self.params.keys():
            self.epochs = self.params["epochs"]

        if "prediction_batch_size" in self.params.keys():
            self.prediction_batch_size = self.params["prediction_batch_size"]

        if "monitor" in self.params.keys():
            self.monitor = self.params["monitor"]

        if "monitor_regime" in self.params.keys():
            self.monitor_regime = self.params["monitor_regime"]

        if "ckpt_freq" in self.params.keys():
            self.ckpt_freq = self.params["ckpt_freq"]

        if "ckpt_save_best_only" in self.params.keys():
            self.ckpt_save_best_only = self.params["ckpt_save_best_only"]

        if "ckpt_path" in self.params.keys():
            self.ckpt_path = self.params["ckpt_path"]

        if "use_early_stopping" in self.params.keys():
            self.use_early_stopping = self.params["use_early_stopping"]

        if "early_stopping_patience" in self.params.keys():
            self.early_stopping_patience = self.params["early_stopping_patience"]

        if "use_reduce_lr_on_plateau" in self.params.keys():
            self.use_reduce_lr_on_plateau = self.params["use_reduce_lr_on_plateau"]

        if "reduce_lr_on_plateau_patience" in self.params.keys():
            self.reduce_lr_on_plateau_patience = self.params["reduce_lr_on_plateau_patience"]

        if "reduce_lr_on_plateau_factor" in self.params.keys():
            self.reduce_lr_on_plateau_factor = self.params["reduce_lr_on_plateau_factor"]

        if "reduce_lr_on_plateau_min" in self.params.keys():
            self.reduce_lr_on_plateau_min = self.params["reduce_lr_on_plateau_min"]

    def set_metrics_collection(self, metrics_collection):

        """
        This method sets metrics collection that will be available for model training
        :param metrics_collection: dictionary of available metrics
        :return: None
        """

        self.metrics_collection = metrics_collection

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

    def create_checkpoint(self):
        self.ckpt = tf.train.Checkpoint(self.model)

    def build(self):
        pass

    def restore(self, ckpt_path='', latest=False):

        if latest:
            path_to_restore = tf.train.latest_checkpoint
        else:
            path_to_restore = ckpt_path

        self.logger.log(f'Restore checkpoint {path_to_restore}')
        self.ckpt.restore(path_to_restore)

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
        :return: list of metrics and metrics number
        """

        metrics_fns = list()
        metrics_num = 0
        for metrics_args_for_output in self.metrics_args:

            metrics_fns_for_output = list()
            for metric_args in metrics_args_for_output:
                metric_class = self.metrics_collection[metric_args["metric_name"]]
                metric_obj = metric_class()
                metric_obj.parse_args(params=metric_args)
                metric_fn = metric_obj.get_metric()
                metrics_fns_for_output.append(metric_fn)

                metrics_num += 1

            metrics_fns.append(metrics_fns_for_output)

        return metrics_fns, metrics_num

    def _checkpoint(self, epoch):

        if epoch % self.ckpt_freq != 0:
            return

        if (self.monitor_regime == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.monitor_regime == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.ckpt_best_epoch = epoch
            self.ckpt.save(self.ckpt_path)
            self.logger.log(f'Checkpoint saved for epoch {epoch}, {self.monitor} = {self.monitor_cur_val:.4f}')
        elif not self.ckpt_save_best_only:
            self.ckpt.save(self.ckpt_path)
            self.logger.log(f'Checkpoint saved for epoch {epoch}')

    def _stop_early(self):

        if (self.monitor_regime == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.monitor_regime == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.early_stopping_cnt = 1
            self.logger.log(f'Early stopping count reset: {self.early_stopping_cnt - 1}')
            return False
        elif self.early_stopping_cnt < self.early_stopping_patience:
            self.early_stopping_cnt += 1
            self.logger.log(f'Early stopping count incremented: {self.early_stopping_cnt - 1}')
            return False
        else:
            self.logger.log(f'Early stopping count is {self.early_stopping_cnt} and equal to early stopping patience')
            self.logger.log(f'Training is stopped early')
            return True

    def _reduce_lr_on_plateau(self, cur_lr):

        if (self.monitor_regime == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.monitor_regime == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.reduce_lr_on_plateau_cnt = 1
            self.logger.log(f'Reduce learning rate on plateau count reset: {self.reduce_lr_on_plateau_cnt - 1}')
            return cur_lr
        elif self.reduce_lr_on_plateau_cnt < self.reduce_lr_on_plateau_patience:
            self.reduce_lr_on_plateau_cnt += 1
            self.logger.log(f'Reduce learning rate on plateau count incremented: {self.reduce_lr_on_plateau_cnt - 1}')
            return cur_lr
        elif cur_lr > self.reduce_lr_on_plateau_min:
            reduced_lr = cur_lr * self.reduce_lr_on_plateau_factor
            self.reduce_lr_on_plateau_cnt = 1
            self.logger.log(f'Learning rate reduced: {reduced_lr}')
            self.logger.log(f'Reduce learning rate on plateau count reset: {self.reduce_lr_on_plateau_cnt - 1}')
            return reduced_lr
        else:
            return cur_lr

    def _update_monitor_best_value(self):

        if (self.monitor_regime == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.monitor_regime == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.monitor_best_val = self.monitor_cur_val

    @staticmethod
    def _log_scalar_to_mlops_server(mlops_task, plot_name, curve_name, val, iteration_num):

        if mlops_task is not None:
            logger = mlops_task.get_logger()
            logger.report_scalar(plot_name, curve_name, val, iteration=iteration_num)

    def fit(self, fold=0, mlops_task=None):

        # Instantiate optimizer
        optimizer = self._get_optimizer()

        # Get loss functions
        loss_fns, loss_weights = self._get_losses()

        # Get metrics
        metric_fns, metrics_num = self._get_metrics()

        # Instantiate epoch history lists
        lr_epoch_history = list()
        training_epoch_loss_history = tf.zeros([0, len(self.loss_args)])
        val_epoch_loss_history = tf.zeros([0, len(self.loss_args)])
        training_epoch_metrics_history = tf.zeros([0, metrics_num])
        val_epoch_metrics_history = tf.zeros([0, metrics_num])

        # Initialize monitor value
        if self.monitor_regime == 'min':
            self.monitor_best_val = np.inf
        elif self.monitor_regime == 'max':
            self.monitor_best_val = -np.inf
        else:
            self.logger.log(f'Unknown monitoring regime: {self.monitor_regime}, using min regime')
            self.monitor_regime = 'min'
            self.monitor_best_val = np.inf

        self.monitor_cur_val = self.monitor_best_val

        # Initialize patience count
        self.early_stopping_cnt = 1
        self.reduce_lr_on_plateau_cnt = 1

        # TODO conditional @tf.function
        def _step(data, labels, cur_epoch, training):

            with tf.GradientTape() as gradient_tape:

                # Forward pass
                model_output = self.model(data, training=training)

                # Calculate loss
                batch_loss = 0
                batch_loss_list = list()

                if len(self.loss_args) == 1:
                    loss = loss_fns[0](tf.cast(tf.convert_to_tensor(labels), dtype=tf.float32), model_output, epoch_num=cur_epoch)
                    batch_loss_list.append(loss)
                    batch_loss += loss * loss_weights[0]
                else:
                    for int_loss_idx, _ in enumerate(self.loss_args):
                        loss = loss_fns[int_loss_idx](labels[int_loss_idx], model_output[int_loss_idx], epoch_num=cur_epoch)
                        batch_loss_list.append(loss)
                        batch_loss += loss * loss_weights[int_loss_idx]

                # Calculate metrics
                batch_metric_list = list()

                if len(self.metrics_args) == 1:
                    for metric_fn in metric_fns[0]:
                        metric = metric_fn(tf.cast(tf.convert_to_tensor(labels), dtype=tf.float32), model_output, epoch_num=cur_epoch)
                        batch_metric_list.append(metric)

                else:
                    for int_output_idx, metric_fns_for_output in metric_fns:
                        for metric_fn in metric_fns_for_output:
                            metric = metric_fn(tf.cast(tf.convert_to_tensor(labels[int_output_idx]), dtype=tf.float32), model_output[int_output_idx], epoch_num=cur_epoch)
                            batch_metric_list.append(metric)

            # Backward pass
            if training:
                gradients = gradient_tape.gradient(batch_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            return batch_loss_list, batch_metric_list

        # Epochs loop
        for epoch in range(self.epochs):

            self.logger.log(f'\nEpoch {epoch}')

            self.logger.log(f'Learning rate = {optimizer.learning_rate.numpy()}')
            lr_epoch_history.append(optimizer.learning_rate)

            # Log to MLOps Server
            self._log_scalar_to_mlops_server(mlops_task, f'learning_rate', f'lr_f{fold}', optimizer.learning_rate.numpy(), epoch)

            # Instantiate batch history lists
            batch_loss_history = tf.zeros([0, len(self.loss_args)])
            batch_metrics_history = tf.zeros([0, metrics_num])

            # Training mini-batches loop
            for batch_data, batch_labels in tqdm(self.train_sequence, desc='Training'):
                loss_list, metric_list = _step(batch_data, batch_labels, epoch, training=True)
                batch_loss_history = tf.concat([batch_loss_history, tf.reshape(loss_list, shape=(1, len(self.loss_args)))], axis=0)
                batch_metrics_history = tf.concat([batch_metrics_history, tf.reshape(metric_list, shape=(1, metrics_num))], axis=0)

            # Apply actions on training data on end of the epoch
            self.train_sequence.on_epoch_end()

            # Calculate and log mean training loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_args in enumerate(self.loss_args):
                batch_mean_single_loss = tf.math.reduce_mean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                self.logger.log(f'Training loss for output {loss_idx} - {loss_args["metric_name"]}: {batch_mean_single_loss.numpy():.4f}')

                # Log to MLOps server
                self._log_scalar_to_mlops_server(mlops_task, f'loss_for_output_{loss_idx}', f'train_{loss_args["metric_name"]}_f{fold}', batch_mean_single_loss.numpy(), epoch)

            training_epoch_loss_history = tf.concat([training_epoch_loss_history, tf.reshape(epoch_loss_list, shape=(1, len(self.loss_args)))], axis=0)

            # Calculate and log mean training metrics for epoch
            epoch_metric_list = list()
            for output_idx, metrics_args_for_output in enumerate(self.metrics_args):
                for metric_idx, metric_args in enumerate(metrics_args_for_output):
                    batch_mean_single_metric = tf.math.reduce_mean(batch_metrics_history[:, output_idx + metric_idx])
                    epoch_metric_list.append(batch_mean_single_metric)

                    self.logger.log(f'Training metric {metric_idx} for output {output_idx} - {metric_args["metric_name"]}: {batch_mean_single_metric.numpy():.4f}')

                    # Log to MLOps server
                    self._log_scalar_to_mlops_server(mlops_task, f'metric_{metric_idx}_for_output_{output_idx}', f'train_{metric_args["metric_name"]}_f{fold}', batch_mean_single_metric.numpy(), epoch)

            training_epoch_metrics_history = tf.concat([training_epoch_metrics_history, tf.reshape(epoch_metric_list, shape=(1, metrics_num))], axis=0)

            # Zero batch loss and metric history after training mini-batches loop
            batch_loss_history = tf.zeros([0, len(self.loss_args)])
            batch_metrics_history = tf.zeros([0, metrics_num])

            # Validation mini-batches loop
            for batch_data, batch_labels in tqdm(self.val_sequence, desc="Validation"):
                loss_list, metric_list = _step(batch_data, batch_labels, epoch, training=False)
                batch_loss_history = tf.concat([batch_loss_history, tf.reshape(loss_list, shape=(1, len(self.loss_args)))], axis=0)
                batch_metrics_history = tf.concat([batch_metrics_history, tf.reshape(metric_list, shape=(1, metrics_num))], axis=0)

            # Apply actions on validation data on end of the epoch
            self.val_sequence.on_epoch_end()

            # Calculate and log mean validation loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_args in enumerate(self.loss_args):
                batch_mean_single_loss = tf.math.reduce_mean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                if f'{loss_args["metric_name"]}_loss_{loss_idx}' == self.monitor:
                    self.monitor_cur_val = batch_mean_single_loss.numpy()

                self.logger.log(f'Validation loss for output {loss_idx} - {loss_args["metric_name"]}: {batch_mean_single_loss.numpy():.4f}')

                # Log to MLOps server
                self._log_scalar_to_mlops_server(mlops_task, f'loss_for_output_{loss_idx}', f'val_{loss_args["metric_name"]}_f{fold}', batch_mean_single_loss.numpy(), epoch)

            val_epoch_loss_history = tf.concat([val_epoch_loss_history, tf.reshape(epoch_loss_list, shape=(1, len(self.loss_args)))], axis=0)

            # Calculate and log mean validation metrics for epoch
            epoch_metric_list = list()
            for output_idx, metrics_args_for_output in enumerate(self.metrics_args):
                for metric_idx, metric_args in enumerate(metrics_args_for_output):
                    batch_mean_single_metric = tf.math.reduce_mean(batch_metrics_history[:, output_idx + metric_idx])
                    epoch_metric_list.append(batch_mean_single_metric)

                    if f'{metric_args["metric_name"]}_metric_{output_idx}_{metric_idx}' == self.monitor:
                        self.monitor_cur_val = batch_mean_single_metric.numpy()

                    self.logger.log(f'Validation metric {metric_idx} for output {output_idx} - {metric_args["metric_name"]}: {batch_mean_single_metric.numpy():.4f}')

                    # Log to MLOps server
                    self._log_scalar_to_mlops_server(mlops_task, f'metric_{metric_idx}_for_output_{output_idx}', f'val_{metric_args["metric_name"]}_f{fold}', batch_mean_single_metric.numpy(), epoch)

            val_epoch_metrics_history = tf.concat([val_epoch_metrics_history, tf.reshape(epoch_metric_list, shape=(1, metrics_num))], axis=0)

            # Save checkpoint
            self._checkpoint(epoch)

            # Stop training early
            if self.use_early_stopping and self._stop_early():
                break

            # Reduce learning rate on plateau
            if self.use_reduce_lr_on_plateau:
                optimizer.learning_rate = self._reduce_lr_on_plateau(optimizer.learning_rate.numpy())

            # Update monitor best value for next epoch
            self._update_monitor_best_value()

    def predict(self, data):

        predictions = self.model.predict(np.array(data), batch_size=self.prediction_batch_size, verbose=1)
        return predictions

    def summary(self):
        self.logger.log("Model Architecture", level=LogLevels.DEBUG, console=True)
        self.model.summary(print_fn=lambda x: self.logger.log(x, level=LogLevels.DEBUG, console=True))
