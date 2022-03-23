import abc
import numpy as np
from loguru import logger

from tqdm import tqdm


default_settings = {
    'epochs': None,
    'monitor': 'not_defined',
    'monitor_regime': 'not_defined',
    'ckpt_freq': 1,
    'ckpt_save_best_only': True,
    'use_early_stopping': False,
    'early_stopping_patience': 10,
    'use_reduce_lr_on_plateau': False,
    'reduce_lr_on_plateau_patience': 3,
    'reduce_lr_on_plateau_factor': 0.8,
    'reduce_lr_on_plateau_min': 1e-6
}


class BaseSingleModelApp(abc.ABC):

    """
    This class defines common methods to all Tensorflow models
    """

    def __init__(
            self,
            settings,
            model,
            optimizer,
            scheduler,
            loss_fns,
            loss_weights,
            loss_names,
            metric_fns,
            metric_names,
            metric_num
            ):
        
        """
        This method initializes parameters
        :return: None 
        """
        
        self.settings = settings

        self.model = model
        self.loss_fns = loss_fns
        self.loss_names = loss_names
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_fns = metric_fns
        self.metric_names = metric_names
        self.metric_num = metric_num

        self.monitor_cur_val = 0
        self.monitor_best_val = 0
        self.ckpt_best_epoch = 0
        self.early_stopping_cnt = 0
        self.reduce_lr_on_plateau_cnt = 0

    @abc.abstractmethod
    def _get_latest_ckpt(self):
        pass

    @abc.abstractmethod
    def _restore_ckpt(self, ckpt_path):
        pass

    @abc.abstractmethod
    def _save_ckpt(self, ckpt_path):
        pass

    @abc.abstractmethod
    def _get_current_lr(self):
        pass

    def restore(self, ckpt_path='', latest=False):

        if latest:
            path_to_restore = self._get_latest_ckpt()
        else:
            path_to_restore = ckpt_path

        logger.info(f'Restore checkpoint {path_to_restore}')
        self._restore_ckpt(path_to_restore)

    def _checkpoint(self, epoch, ckpt_path):

        if epoch % self.settings['ckpt_freq'] != 0:
            return

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.ckpt_best_epoch = epoch
            self._save_ckpt(ckpt_path)
            logger.info(f'Checkpoint saved for epoch {epoch}, {self.settings["monitor"]} = {self.monitor_cur_val:.4f}')
        elif not self.settings['ckpt_save_best_only']:
            self._save_ckpt(ckpt_path)
            logger.info(f'Checkpoint saved for epoch {epoch}')

    def _stop_early(self):

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.early_stopping_cnt = 1
            logger.info(f'Early stopping count reset: {self.early_stopping_cnt - 1}')
            return False
        elif self.early_stopping_cnt < self.settings['early_stopping_patience']:
            self.early_stopping_cnt += 1
            logger.info(f'Early stopping count incremented: {self.early_stopping_cnt - 1}')
            return False
        else:
            logger.info(f'Early stopping count is {self.early_stopping_cnt} and equal to early stopping patience')
            logger.info(f'Training is stopped early')
            return True

    def _reduce_lr_on_plateau(self, cur_lr):

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.reduce_lr_on_plateau_cnt = 1
            logger.info(f'Reduce learning rate on plateau count reset: {self.reduce_lr_on_plateau_cnt - 1}')
            return cur_lr
        elif self.reduce_lr_on_plateau_cnt < self.settings['reduce_lr_on_plateau_patience']:
            self.reduce_lr_on_plateau_cnt += 1
            logger.info(f'Reduce learning rate on plateau count incremented: {self.reduce_lr_on_plateau_cnt - 1}')
            return cur_lr
        elif cur_lr > self.settings['reduce_lr_on_plateau_min']:
            reduced_lr = cur_lr * self.settings['reduce_lr_on_plateau_factor']
            self.reduce_lr_on_plateau_cnt = 1
            logger.info(f'Learning rate reduced: {reduced_lr}')
            logger.info(f'Reduce learning rate on plateau count reset: {self.reduce_lr_on_plateau_cnt - 1}')
            return reduced_lr
        else:
            return cur_lr

    def _update_monitor_best_value(self):

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.monitor_best_val = self.monitor_cur_val

    @staticmethod
    def _log_scalar_to_mlops_server(mlops_task, plot_name, curve_name, val, iteration_num):

        if mlops_task is not None:
            mlops_logger = mlops_task.get_logger()
            mlops_logger.report_scalar(plot_name, curve_name, val, iteration=iteration_num)

    @abc.abstractmethod
    def _step(self, data, labels, training):
        pass

    def fit(self, train_data_loader, val_data_loader, ckpt_path, fold=0, mlops_task=None):

        # Instantiate epoch history lists
        lr_epoch_history = list()
        training_epoch_loss_history = np.zeros([0, len(self.loss_fns)])
        val_epoch_loss_history = np.zeros([0, len(self.loss_fns)])
        training_epoch_metrics_history = np.zeros([0, self.metric_num])
        val_epoch_metrics_history = np.zeros([0, self.metric_num])

        # Initialize monitor value
        if self.settings['monitor_regime'] == 'min':
            self.monitor_best_val = np.inf
        elif self.settings['monitor_regime'] == 'max':
            self.monitor_best_val = -np.inf
        else:
            logger.info(f'Unknown monitoring regime: {self.settings["monitor_regime"]}, using min regime')
            self.settings['monitor_regime'] = 'min'
            self.monitor_best_val = np.inf

        self.monitor_cur_val = self.monitor_best_val

        # Initialize patience count
        self.early_stopping_cnt = 1
        self.reduce_lr_on_plateau_cnt = 1

        # Initialize best epoch value
        self.ckpt_best_epoch = 0

        # Epochs loop
        for epoch in range(self.settings['epochs']):

            logger.info(f'\nEpoch {epoch}')

            current_lr = self._get_current_lr()
            logger.info(f'Learning rate = {current_lr}')
            lr_epoch_history.append(current_lr)

            # Log to MLOps Server
            self._log_scalar_to_mlops_server(mlops_task, f'learning_rate', f'lr_f{fold}', current_lr, epoch)

            # Instantiate batch history lists
            batch_loss_history = np.zeros([0, len(self.loss_fns)])
            batch_metrics_history = np.zeros([0, self.metric_num])

            # Training mini-batches loop
            for batch_data, batch_labels, _ in tqdm(train_data_loader, desc='Training'):
                loss_list, metric_list = self._step(batch_data, batch_labels, training=True)
                batch_loss_history = np.concatenate([batch_loss_history, np.reshape(loss_list, newshape=(1, len(self.loss_fns)))], axis=0)
                batch_metrics_history = np.concatenate([batch_metrics_history, np.reshape(metric_list, newshape=(1, self.metric_num))], axis=0)

            # Calculate and log mean training loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_name in enumerate(self.loss_names):
                batch_mean_single_loss = np.nanmean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                logger.info(f'Training loss for output {loss_idx} - {loss_name}: {batch_mean_single_loss:.4f}')

                # Log to MLOps server
                self._log_scalar_to_mlops_server(mlops_task, f'loss_for_output_{loss_idx}', f'train_{loss_name}_f{fold}', batch_mean_single_loss, epoch)

            training_epoch_loss_history = np.concatenate([training_epoch_loss_history, np.reshape(epoch_loss_list, newshape=(1, len(self.loss_fns)))], axis=0)

            # Calculate and log mean training metrics for epoch
            epoch_metric_list = list()
            for output_idx, metric_names_for_output in enumerate(self.metric_names):
                for metric_idx, metric_name in enumerate(metric_names_for_output):
                    batch_mean_single_metric = np.nanmean(batch_metrics_history[:, output_idx + metric_idx])
                    epoch_metric_list.append(batch_mean_single_metric)

                    logger.info(f'Training metric {metric_idx} for output {output_idx} - {metric_name}: {batch_mean_single_metric:.4f}')

                    # Log to MLOps server
                    self._log_scalar_to_mlops_server(mlops_task, f'metric_{metric_idx}_for_output_{output_idx}', f'train_{metric_name}_f{fold}', batch_mean_single_metric, epoch)

            training_epoch_metrics_history = np.concatenate([training_epoch_metrics_history, np.reshape(epoch_metric_list, newshape=(1, self.metric_num))], axis=0)

            # Zero batch loss and metric history after training mini-batches loop
            batch_loss_history = np.zeros([0, len(self.loss_fns)])
            batch_metrics_history = np.zeros([0, self.metric_num])

            # Validation mini-batches loop
            for batch_data, batch_labels, _ in tqdm(val_data_loader, desc="Validation"):
                loss_list, metric_list = self._step(batch_data, batch_labels, training=False)
                batch_loss_history = np.concatenate([batch_loss_history, np.reshape(loss_list, newshape=(1, len(self.loss_fns)))], axis=0)
                batch_metrics_history = np.concatenate([batch_metrics_history, np.reshape(metric_list, newshape=(1, self.metric_num))], axis=0)

            # Calculate and log mean validation loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_name in enumerate(self.loss_names):
                batch_mean_single_loss = np.nanmean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                if f'{loss_name}_loss_{loss_idx}' == self.settings['monitor']:
                    self.monitor_cur_val = batch_mean_single_loss

                logger.info(f'Validation loss for output {loss_idx} - {loss_name}: {batch_mean_single_loss:.4f}')

                # Log to MLOps server
                self._log_scalar_to_mlops_server(mlops_task, f'loss_for_output_{loss_idx}', f'val_{loss_name}_f{fold}', batch_mean_single_loss, epoch)

            val_epoch_loss_history = np.concatenate([val_epoch_loss_history, np.reshape(epoch_loss_list, newshape=(1, len(self.loss_fns)))], axis=0)

            # Calculate and log mean validation metrics for epoch
            epoch_metric_list = list()
            for output_idx, metric_names_for_output in enumerate(self.metric_names):
                for metric_idx, metric_name in enumerate(metric_names_for_output):
                    batch_mean_single_metric = np.nanmean(batch_metrics_history[:, output_idx + metric_idx])
                    epoch_metric_list.append(batch_mean_single_metric)

                    if f'{metric_name}_metric_{output_idx}_{metric_idx}' == self.settings['monitor']:
                        self.monitor_cur_val = batch_mean_single_metric

                    logger.info(f'Validation metric {metric_idx} for output {output_idx} - {metric_name}: {batch_mean_single_metric:.4f}')

                    # Log to MLOps server
                    self._log_scalar_to_mlops_server(mlops_task, f'metric_{metric_idx}_for_output_{output_idx}', f'val_{metric_name}_f{fold}', batch_mean_single_metric, epoch)

            val_epoch_metrics_history = np.concatenate([val_epoch_metrics_history, np.reshape(epoch_metric_list, newshape=(1, self.metric_num))], axis=0)

            # Save checkpoint
            self._checkpoint(epoch, ckpt_path)

            # Stop training early
            if self.settings['use_early_stopping'] and self._stop_early():
                break

            # Reduce learning rate on plateau
            if self.settings['use_reduce_lr_on_plateau']:
                self.optimizer.learning_rate = self._reduce_lr_on_plateau(self.optimizer.learning_rate.numpy())

            # Adjust learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Update monitor best value for next epoch
            self._update_monitor_best_value()

    @abc.abstractmethod
    def predict(self, data):
        # return predictions
        pass

    @abc.abstractmethod
    def summary(self):
        pass
