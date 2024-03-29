import torch
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
    'plateau_patience': 3,
}


class Trainer(object):

    """
    This class defines training logic for the application.
    It assumes that application implements interface defined in 'BaseApp'.
    """

    def __init__(
            self,
            settings
            ):
        
        """
        This method initializes parameters.
        :param settings: Dictionary that contains configuration parameters.
        :return: None 
        """
        
        self.settings = settings

        self.monitor_cur_val = 0
        self.monitor_best_val = 0
        self.ckpt_best_epoch = 0
        self.early_stopping_cnt = 0
        self.plateau_cnt = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_checkpoint(self, app, epoch, ckpt_path):

        if epoch % self.settings['ckpt_freq'] != 0:
            return

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.ckpt_best_epoch = epoch
            app.save_ckpt(ckpt_path)
            logger.info(f'Checkpoint saved for epoch {epoch}, {self.settings["monitor"]} = {self.monitor_cur_val:.4f}')
        elif not self.settings['ckpt_save_best_only']:
            app.save_ckpt(ckpt_path)
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

    def _is_plateau(self):

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.plateau_cnt = 1
            logger.info(f'Plateau count reset: {self.plateau_cnt - 1}')
            return False
        elif self.plateau_cnt < self.settings['plateau_patience']:
            self.plateau_cnt += 1
            logger.info(f'Plateau count incremented: {self.plateau_cnt - 1}')
            return False
        else:
            self.plateau_cnt = 1
            logger.info(f'Plateau detected')
            logger.info(f'Plateau count reset: {self.plateau_cnt - 1}')
            return True

    def update_monitor_best_value(self):

        if (self.settings['monitor_regime'] == "min" and self.monitor_cur_val < self.monitor_best_val) or \
           (self.settings['monitor_regime'] == "max" and self.monitor_cur_val > self.monitor_best_val):
            self.monitor_best_val = self.monitor_cur_val

    def fit(self, app, train_data_loader, val_data_loader, ckpt_path, fold=0, mlops_task=None):

        # Instantiate epoch history lists
        lr_epoch_history = np.zeros([0, len(app.get_lr())])
        training_epoch_loss_history = np.zeros([0, app.losses_num])
        val_epoch_loss_history = np.zeros([0, app.losses_num])
        training_epoch_metrics_history = np.zeros([0, app.metrics_num])
        val_epoch_metrics_history = np.zeros([0, app.metrics_num])

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

            # Do application specific actions on epoch start
            app.on_epoch_start()

            # Log learning rate/s
            current_lr = app.get_lr()
            lr_list = list()
            for lr_idx, (key, value) in enumerate(current_lr.items()):
                logger.info(f'Learning rate: {key} = {value}')
                lr_list.append(current_lr[key])

                # Log to MLOps Server
                mlops_task.log_scalar_to_mlops_server(f'Learning rate', f'{key}_f{fold}', current_lr[key], epoch)

            lr_epoch_history = np.concatenate(
                [lr_epoch_history, np.reshape(lr_list, newshape=(1, len(app.get_lr())))],
                axis=0
            )

            # Instantiate batch history lists
            batch_loss_history = np.zeros([0, app.losses_num])
            batch_metrics_history = np.zeros([0, app.metrics_num])

            # Training mini-batches loop
            iterations_per_epoch = len(train_data_loader)
            for cur_iteration, batch_data in enumerate(tqdm(train_data_loader, desc='Training')):
                loss_list, metric_list = app.training_step(batch_data, epoch, cur_iteration, iterations_per_epoch)
                batch_loss_history = np.concatenate([batch_loss_history, np.reshape(loss_list, newshape=(1, app.losses_num))], axis=0)
                batch_metrics_history = np.concatenate([batch_metrics_history, np.reshape(metric_list, newshape=(1, app.metrics_num))], axis=0)

            # Calculate and log mean training loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_name in enumerate(app.losses_names):
                batch_mean_single_loss = np.nanmean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                logger.info(f'Training loss {loss_name}: {batch_mean_single_loss:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'loss_{loss_name}', f'train_{loss_name}_f{fold}', batch_mean_single_loss, epoch)

            training_epoch_loss_history = np.concatenate([training_epoch_loss_history, np.reshape(epoch_loss_list, newshape=(1, app.losses_num))], axis=0)

            # Calculate and log mean training metrics for epoch
            epoch_metric_list = list()
            for metric_idx, metric_name in enumerate(app.metrics_names):
                batch_mean_single_metric = np.nanmean(batch_metrics_history[:, metric_idx])
                epoch_metric_list.append(batch_mean_single_metric)

                logger.info(f'Training metric {metric_name}: {batch_mean_single_metric:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'metric_{metric_name}', f'train_{metric_name}_f{fold}', batch_mean_single_metric, epoch)

            training_epoch_metrics_history = np.concatenate([training_epoch_metrics_history, np.reshape(epoch_metric_list, newshape=(1, app.metrics_num))], axis=0)

            # Zero batch loss and metric history after training mini-batches loop
            batch_loss_history = np.zeros([0, app.losses_num])
            batch_metrics_history = np.zeros([0, app.metrics_num])

            # Validation mini-batches loop
            iterations_per_epoch = len(val_data_loader)
            for cur_iteration, batch_data in enumerate(tqdm(val_data_loader, desc="Validation")):
                loss_list, metric_list = app.validation_step(batch_data, epoch, cur_iteration, iterations_per_epoch)
                batch_loss_history = np.concatenate([batch_loss_history, np.reshape(loss_list, newshape=(1, app.losses_num))], axis=0)
                batch_metrics_history = np.concatenate([batch_metrics_history, np.reshape(metric_list, newshape=(1, app.metrics_num))], axis=0)

            # Calculate and log mean validation loss for epoch
            epoch_loss_list = list()
            for loss_idx, loss_name in enumerate(app.losses_names):
                batch_mean_single_loss = np.nanmean(batch_loss_history[:, loss_idx])
                epoch_loss_list.append(batch_mean_single_loss)

                if f'{loss_name}' == self.settings['monitor']:
                    self.monitor_cur_val = batch_mean_single_loss

                logger.info(f'Validation loss {loss_name}: {batch_mean_single_loss:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'loss_{loss_name}', f'val_{loss_name}_f{fold}', batch_mean_single_loss, epoch)

            val_epoch_loss_history = np.concatenate([val_epoch_loss_history, np.reshape(epoch_loss_list, newshape=(1, app.losses_num))], axis=0)

            # Calculate and log mean validation metrics for epoch
            epoch_metric_list = list()
            for metric_idx, metric_name in enumerate(app.metrics_names):
                batch_mean_single_metric = np.nanmean(batch_metrics_history[:, metric_idx])
                epoch_metric_list.append(batch_mean_single_metric)

                if f'{metric_name}' == self.settings['monitor']:
                    self.monitor_cur_val = batch_mean_single_metric

                logger.info(f'Validation metric {metric_name}: {batch_mean_single_metric:.4f}')

                # Log to MLOps server
                mlops_task.log_scalar_to_mlops_server(f'metric_{metric_name}', f'val_{metric_name}_f{fold}', batch_mean_single_metric, epoch)

            val_epoch_metrics_history = np.concatenate([val_epoch_metrics_history, np.reshape(epoch_metric_list, newshape=(1, app.metrics_num))], axis=0)

            # Save checkpoint
            self.create_checkpoint(app, epoch, ckpt_path)

            # Stop training early
            if self.settings['use_early_stopping'] and self.stop_early():
                break

            # Do application specific actions on epoch end
            is_plateau = self._is_plateau()
            app.on_epoch_end(is_plateau)

            # Update monitor best value for next epoch
            self.update_monitor_best_value()
