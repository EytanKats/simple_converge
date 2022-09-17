import abc
import numpy as np
from tqdm import tqdm
from loguru import logger


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


class BaseApp(abc.ABC):

    """
    This class defines common methods to all Tensorflow models
    """

    def __init__(
            self,
            settings,
            losses_fns,
            losses_names,
            metrics_fns,
            metrics_names
            ):
        
        """
        This method initializes parameters.
        :param settings: Dictionary that contains configuration parameters.
        :param losses_names: List that contains names of loss functions names in the same order as returned by step method.
        These names will be used for logging.
        :param metrics_names: List that contains names of metrics names in the same order as returned by step method.
        These names will be used for logging.
        :return: None 
        """
        
        self.settings = settings

        self.losses_fns = losses_fns
        self.losses_names = losses_names
        self.losses_num = len(losses_names)
        self.metrics_fns = metrics_fns
        self.metrics_names = metrics_names
        self.metrics_num = len(metrics_names)

        self.monitor_cur_val = 0
        self.monitor_best_val = 0
        self.ckpt_best_epoch = 0
        self.early_stopping_cnt = 0
        self.reduce_lr_on_plateau_cnt = 0

    @abc.abstractmethod
    def save_ckpt(self, ckpt_path):
        pass

    @abc.abstractmethod
    def restore_ckpt(self, ckpt_path=''):
        pass

    @abc.abstractmethod
    def get_lr(self):
        pass

    @abc.abstractmethod
    def set_lr(self, lr):
        pass

    @abc.abstractmethod
    def apply_scheduler(self):
        pass

    def create_checkpoint(self, epoch, ckpt_path):

        if epoch % self.settings['ckpt_freq'] != 0:
            return

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.ckpt_best_epoch = epoch
            self.save_ckpt(ckpt_path)
            logger.info(f'Checkpoint saved for epoch {epoch}, {self.settings["monitor"]} = {self.monitor_cur_val:.4f}')
        elif not self.settings['ckpt_save_best_only']:
            self.save_ckpt(ckpt_path)
            logger.info(f'Checkpoint saved for epoch {epoch}')

    def stop_early(self):

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

    def reduce_lr_on_plateau(self, cur_lr):

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

    def update_monitor_best_value(self):

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.monitor_best_val = self.monitor_cur_val

    @abc.abstractmethod
    def training_step(self, data, epoch):
        pass

    @abc.abstractmethod
    def validation_step(self, data, epoch):
        pass

    def fit(self, train_data_loader, val_data_loader, ckpt_path, fold=0, mlops_task=None):

        # Instantiate epoch history lists
        lr_epoch_history = list()
        training_epoch_loss_history = np.zeros([0, self.losses_num])
        val_epoch_loss_history = np.zeros([0, self.losses_num])
        training_epoch_metrics_history = np.zeros([0, self.metrics_num])
        val_epoch_metrics_history = np.zeros([0, self.metrics_num])

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

            current_lr = self.get_lr()
            logger.info(f'Learning rate = {current_lr}')
            lr_epoch_history.append(current_lr)

            # Log to MLOps Server
            mlops_task.log_scalar_to_mlops_server(f'learning_rate', f'lr_f{fold}', current_lr, epoch)

            # Instantiate batch history lists
            batch_loss_history = np.zeros([0, self.losses_num])
            batch_metrics_history = np.zeros([0, self.metrics_num])

            # Training mini-batches loop
            for batch_data in tqdm(train_data_loader, desc='Training'):
                loss_list, metric_list = self.training_step(batch_data, epoch)
                batch_loss_history = np.concatenate([batch_loss_history, np.reshape(loss_list, newshape=(1, self.losses_num))], axis=0)
                batch_metrics_history = np.concatenate([batch_metrics_history, np.reshape(metric_list, newshape=(1, self.metrics_num))], axis=0)

            # Calculate and log mean training loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_name in enumerate(self.losses_names):
                batch_mean_single_loss = np.nanmean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                logger.info(f'Training loss {loss_name}: {batch_mean_single_loss:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'loss_{loss_name}', f'train_{loss_name}_f{fold}', batch_mean_single_loss, epoch)

            training_epoch_loss_history = np.concatenate([training_epoch_loss_history, np.reshape(epoch_loss_list, newshape=(1, self.losses_num))], axis=0)

            # Calculate and log mean training metrics for epoch
            epoch_metric_list = list()
            for metric_idx, metric_name in enumerate(self.metrics_names):
                batch_mean_single_metric = np.nanmean(batch_metrics_history[:, metric_idx])
                epoch_metric_list.append(batch_mean_single_metric)

                logger.info(f'Training metric {metric_name}: {batch_mean_single_metric:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'metric_{metric_name}', f'train_{metric_name}_f{fold}', batch_mean_single_metric, epoch)

            training_epoch_metrics_history = np.concatenate([training_epoch_metrics_history, np.reshape(epoch_metric_list, newshape=(1, self.metrics_num))], axis=0)

            # Zero batch loss and metric history after training mini-batches loop
            batch_loss_history = np.zeros([0, self.losses_num])
            batch_metrics_history = np.zeros([0, self.metrics_num])

            # Validation mini-batches loop
            for batch_data in tqdm(val_data_loader, desc="Validation"):
                loss_list, metric_list = self.validation_step(batch_data, epoch)
                batch_loss_history = np.concatenate([batch_loss_history, np.reshape(loss_list, newshape=(1, self.losses_num))], axis=0)
                batch_metrics_history = np.concatenate([batch_metrics_history, np.reshape(metric_list, newshape=(1, self.metrics_num))], axis=0)

            # Calculate and log mean validation loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_name in enumerate(self.losses_names):
                batch_mean_single_loss = np.nanmean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                if f'{loss_name}' == self.settings['monitor']:
                    self.monitor_cur_val = batch_mean_single_loss

                logger.info(f'Validation loss {loss_name}: {batch_mean_single_loss:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'loss_{loss_name}', f'val_{loss_name}_f{fold}', batch_mean_single_loss, epoch)

            val_epoch_loss_history = np.concatenate([val_epoch_loss_history, np.reshape(epoch_loss_list, newshape=(1, self.losses_num))], axis=0)

            # Calculate and log mean validation metrics for epoch
            epoch_metric_list = list()
            for metric_idx, metric_name in enumerate(self.metrics_names):
                batch_mean_single_metric = np.nanmean(batch_metrics_history[:, metric_idx])
                epoch_metric_list.append(batch_mean_single_metric)

                if f'{metric_name}' == self.settings['monitor']:
                    self.monitor_cur_val = batch_mean_single_metric

                logger.info(f'Validation metric {metric_name}: {batch_mean_single_metric:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'metric_{metric_name}', f'val_{metric_name}_f{fold}', batch_mean_single_metric, epoch)

            val_epoch_metrics_history = np.concatenate([val_epoch_metrics_history, np.reshape(epoch_metric_list, newshape=(1, self.metrics_num))], axis=0)

            # Save checkpoint
            self.create_checkpoint(epoch, ckpt_path)

            # Stop training early
            if self.settings['use_early_stopping'] and self.stop_early():
                break

            # Reduce learning rate on plateau
            if self.settings['use_reduce_lr_on_plateau']:
                reduced_lr = self.reduce_lr_on_plateau(self.get_lr())
                self.set_lr(reduced_lr)

            # Adjust learning rate
            self.apply_scheduler()

            # Update monitor best value for next epoch
            self.update_monitor_best_value()

    @abc.abstractmethod
    def predict(self, data):
        # return predictions
        pass

    @abc.abstractmethod
    def summary(self):
        pass
