import torch
from loguru import logger

from simple_converge.apps.BaseApp import BaseApp

default_settings = {
    'use_ema': False,
    'ema_decay': 0.999
}


class MomentumApp(BaseApp):
    """
    This class defines self-supervised application with two encoders:
    first encoder learns through backpropagation,
    second encoder updated as Exponential Moving Average of the first.
    Each of two input views (two different augmentations of the input) propagated through both encoders
    while self-supervised loss function gets outputs corresponding to two different views and applied in symmetric way.
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

        super(MomentumApp, self).__init__(
            settings,
            mlops_task,
            loss_function,
            metric,
        )

        if architecture is not None:
            self.base_encoder = architecture['base_encoder'](settings)
            self.momentum_encoder = architecture['base_encoder'](settings)
            self.predictor = architecture['predictor'](settings)
        else:
            self.base_encoder = None
            self.momentum_encoder = None
            self.predictor = None

        # Initialize parameters of momentum encoder
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        if optimizer is not None:
            # Instantiate optimizer for base encoder and predictor but not for the momentum encoder
            self.optimizer = optimizer(settings, self.base_encoder, self.predictor)
        else:
            self.optimizer = None

        if scheduler is not None:
            self.scheduler = scheduler(settings, self.optimizer)
        else:
            self.scheduler = None

        self.ckpt_cnt = 0
        self.latest_ckpt_path = None

    def restore_ckpt(self, ckpt_path=''):
        if ckpt_path:
            path_to_restore = ckpt_path
        else:
            path_to_restore = self.latest_ckpt_path

        logger.info(f'Restore checkpoint {path_to_restore}')
        checkpoint = torch.load(ckpt_path)
        self.base_encoder.load_state_dict(checkpoint['base_encoder'])
        self.momentum_encoder.load_state_dict(checkpoint['momentum_encoder'])
        self.predictor.load_state_dict(checkpoint['predictor'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_ckpt(self, ckpt_path):

        self.latest_ckpt_path = ckpt_path + '-' + str(self.ckpt_cnt) + '.pth'
        self.ckpt_cnt += 1

        torch.save({
            'base_encoder': self.base_encoder.state_dict(),
            'momentum_encoder': self.momentum_encoder.state_dict(),
            'predictor': self.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        },
            self.latest_ckpt_path
        )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def apply_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def _update_momentum_encoder(self, m):

        # Update parameters of momentum encoder
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def training_step(self, data, epoch):

        self.base_encoder.train()

        view_1 = data[0].to(self.device)
        view_2 = data[1].to(self.device)

        self.optimizer.zero_grad()

        # Compute features of base encoder
        base_features_1 = self.base_encoder(view_1)
        base_features_2 = self.base_encoder(view_2)

        # Compute predictions
        predictions_1 = self.predictor(base_features_1)
        predictions_2 = self.predictor(base_features_2)

        with torch.no_grad():  # disable gradient calculation
            self._update_momentum_encoder(self.ema_decay)  # update the momentum encoder

            # Compute features of momentum encoder
            momentum_features_1 = self.momentum_encoder(view_1)
            momentum_features_2 = self.momentum_encoder(view_2)

        # Calculate loss
        batch_loss_list = list()

        loss = self.losses_fns[0](predictions_1, momentum_features_2)\
               + self.losses_fns[0](predictions_2, momentum_features_1)
        batch_loss_list.append(loss.detach().cpu().numpy())

        # Calculate metrics
        batch_metric_list = list()

        for metric_fn in self.metrics_fns:
            metric = metric_fn(predictions_1, momentum_features_2)\
                     + metric_fn(predictions_2, momentum_features_1)
            batch_metric_list.append(metric.detach().cpu().numpy())

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return batch_loss_list, batch_metric_list

    def validation_step(self, data, epoch):

        self.base_encoder.eval()

        view_1 = data[0].to(self.device)
        view_2 = data[1].to(self.device)

        with torch.no_grad():

            # Compute features of base encoder
            base_features_1 = self.base_encoder(view_1)
            base_features_2 = self.base_encoder(view_2)

            # Compute predictions
            predictions_1 = self.predictor(base_features_1)
            predictions_2 = self.predictor(base_features_2)

            # Compute features of momentum encoder
            momentum_features_1 = self.momentum_encoder(view_1)
            momentum_features_2 = self.momentum_encoder(view_2)

            # Calculate loss
            batch_loss_list = list()

            loss = self.losses_fns[0](predictions_1, momentum_features_2) \
                   + self.losses_fns[0](predictions_2, momentum_features_1)
            batch_loss_list.append(loss.detach().cpu().numpy())

            # Calculate metrics
            batch_metric_list = list()

            for metric_fn in self.metrics_fns:
                metric = metric_fn(predictions_1, momentum_features_2) \
                         + metric_fn(predictions_2, momentum_features_1)
                batch_metric_list.append(metric.detach().cpu().numpy())

        return batch_loss_list, batch_metric_list

    def predict(self, data):

        self.base_encoder.eval()

        view_1 = data[0].to(self.device)
        view_2 = data[1].to(self.device)

        with torch.no_grad():

            # Compute features of base encoder
            base_features_1 = self.base_encoder(view_1)
            base_features_2 = self.base_encoder(view_2)

        base_features_1 = base_features_1.detach().cpu().numpy()
        base_features_2 = base_features_2.detach().cpu().numpy()

        return [base_features_1, base_features_2]

    def summary(self):
        logger.info("Base encoder architecture:")
        print(self.base_encoder)

        logger.info("Momentum encoder architecture:")
        print(self.momentum_encoder)

        logger.info("Predictor architecture:")
        print(self.predictor)
