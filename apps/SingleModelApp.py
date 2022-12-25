import torch
from loguru import logger

from simple_converge.utils.training import EMA
from simple_converge.apps.BaseApp import BaseApp


default_settings = {
    'use_reduce_lr_on_plateau': False,
    'reduce_lr_on_plateau_factor': 0.8,
    'reduce_lr_on_plateau_min': 1e-6,
    'use_ema': False,
    'ema_decay': 0.999
}


class SingleModelApp(BaseApp):

    """
    This class defines single model application.
    """

    def __init__(
        self,
        settings,
        mlops_task,
        architecture,
        loss_function,
        metric,
        scheduler,
        optimizer,
    ):
        
        """
        This method initializes parameters
        :return: None 
        """
        
        super(SingleModelApp, self).__init__(
            settings,
            mlops_task,
            loss_function,
            metric,
        )

        self.model = architecture(settings)

        if optimizer is not None:
            self.optimizer = optimizer(settings, self.model)
        else:
            self.optimizer = None

        if scheduler is not None:
            self.scheduler = scheduler(settings, self.optimizer)
        else:
            self.scheduler = None

        self.ckpt_cnt = 0
        self.latest_ckpt_path = None

        self.ema = EMA(self.model, self.settings['app']['ema_decay'])

    def restore_ckpt(self, ckpt_path=''):
        if ckpt_path:
            path_to_restore = ckpt_path
        else:
            path_to_restore = self.latest_ckpt_path

        logger.info(f'Restore checkpoint {path_to_restore}')
        self.model.load_state_dict(torch.load(ckpt_path))

    def save_ckpt(self, ckpt_path):

        if self.settings['app']['use_ema']:
            self.ema.apply_shadow()

        self.latest_ckpt_path = ckpt_path + '-' + str(self.ckpt_cnt) + '.pth'
        self.ckpt_cnt += 1
        torch.save(self.model.state_dict(), self.latest_ckpt_path)

        if self.settings['app']['use_ema']:
            self.ema.restore()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return {'base_lr': param_group['lr']}

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _apply_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, is_plateau=False):

        # Reduce learning rate on plateau
        if is_plateau and self.settings['app']['use_reduce_lr_on_plateau']:
            cur_lr = self.get_lr()['base_lr']
            if cur_lr > self.settings['app']['reduce_lr_on_plateau_min']:
                reduced_lr = cur_lr * self.settings['reduce_lr_on_plateau_factor']
                self._set_lr(reduced_lr)

        # Apply scheduler
        self._apply_scheduler()

    def training_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        self.model.train()

        input_data = data[0].to(self.device)
        labels = data[1].to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            # Apply model
            model_output = self.model(input_data)

            # Calculate loss
            batch_loss_list = list()

            loss = self.losses_fns[0](model_output, labels)
            batch_loss_list.append(loss.detach().cpu().numpy())

            # Calculate metrics
            batch_metric_list = list()

            for metric_fn in self.metrics_fns:
                metric = metric_fn(model_output, labels)
                batch_metric_list.append(metric.detach().cpu().numpy())

        # Backward pass
        loss.backward()
        self.optimizer.step()

        if self.settings['app']['use_ema']:
            self.ema.update()

        return batch_loss_list, batch_metric_list

    def validation_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        self.model.eval()

        if self.settings['app']['use_ema']:
            self.ema.apply_shadow()

        input_data = data[0].to(self.device)
        labels = data[1].to(self.device)

        with torch.set_grad_enabled(False):

            # Apply model
            model_output = self.model(input_data)

            # Calculate loss
            batch_loss_list = list()

            loss = self.losses_fns[0](model_output, labels)
            batch_loss_list.append(loss.detach().cpu().numpy())

            # Calculate metrics
            batch_metric_list = list()

            for metric_fn in self.metrics_fns:
                metric = metric_fn(model_output, labels)
                batch_metric_list.append(metric.detach().cpu().numpy())

        if self.settings['app']['use_ema']:
            self.ema.restore()

        return batch_loss_list, batch_metric_list

    def predict(self, data):

        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            predictions = self.model(data)
        predictions = predictions.detach().cpu().numpy()
        return predictions

    def summary(self):
        logger.info("Model architecture:")
        print(self.model)
