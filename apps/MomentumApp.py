import math
import torch
from loguru import logger

from simple_converge.apps.BaseApp import BaseApp

default_settings = {
    'momentum': 0.999,
    'warmup_epochs': 10
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
            optimizer
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

        # Create models' architecture
        if architecture is not None:
            self.base_encoder = architecture['encoder'](settings)
            self.momentum_encoder = architecture['encoder'](settings)
            self.base_projector = architecture['projector'](settings)
            self.momentum_projector = architecture['projector'](settings)
            self.predictor = architecture['predictor'](settings)
        else:
            self.base_encoder = None
            self.momentum_encoder = None
            self.base_projector = None
            self.momentum_projector = None
            self.predictor = None

        # Initialize parameters of momentum encoder to be equal to parameters of base encoder
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # prevent update by gradient

        # Initialize parameters of momentum projector to be equal to parameters of base projector
        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # prevent update by gradient

        if optimizer is not None:
            # Instantiate optimizer for base encoder and predictor but not for the momentum encoder
            self.optimizer = optimizer(settings, self.base_encoder, self.base_projector, self.predictor)
        else:
            self.optimizer = None
        self.initial_lr = self.get_lr()['base_lr']  # initial LR is used in 'adjust_lr' method

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
        self.base_projector.load_state_dict(checkpoint['base_projector'])
        self.momentum_projector.load_state_dict(checkpoint['momentum_projector'])
        self.predictor.load_state_dict(checkpoint['predictor'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_ckpt(self, ckpt_path):

        self.latest_ckpt_path = ckpt_path + '-' + str(self.ckpt_cnt) + '.pth'
        self.ckpt_cnt += 1

        torch.save({
            'base_encoder': self.base_encoder.state_dict(),
            'momentum_encoder': self.momentum_encoder.state_dict(),
            'base_projector': self.base_projector.state_dict(),
            'momentum_projector': self.momentum_projector.state_dict(),
            'predictor': self.predictor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        },
            self.latest_ckpt_path
        )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return {'base_lr': param_group['lr']}

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, is_plateau=False):
        pass

    @torch.no_grad()
    def _update_momentum_encoder_and_projector(self, m):

        # Update parameters of momentum encoder
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        # Update parameters of momentum projector
        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def adjust_lr(self, epoch):
        """
        This method decays learning rate with half-cycle cosine after warmup
        """

        if epoch < self.settings['app']['warmup_epochs']:
            lr = self.initial_lr * epoch / self.settings['app']['warmup_epochs']
        else:
            lr = self.initial_lr * 0.5 *\
                 (1. + math.cos(math.pi * (epoch - self.settings['app']['warmup_epochs'])
                                / (self.settings['trainer']['epochs'] - self.settings['app']['warmup_epochs'])))

        self._set_lr(lr)
        return lr

    def apply_momentum_update(self, epoch):

        """
        Adjust momentum based on current epoch
        """

        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / self.settings['trainer']['epochs']))\
            * (1. - self.settings['app']['momentum'])
        self._update_momentum_encoder_and_projector(m)

        return m

    def training_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        self.adjust_lr(epoch + cur_iteration / iterations_per_epoch)
        self.apply_momentum_update(epoch + cur_iteration / iterations_per_epoch)

        self.base_encoder.train()
        self.base_projector.train()
        self.predictor.train()

        # Get two views (different augmentations) of the data
        view_1 = data[0].to(self.device)
        view_2 = data[1].to(self.device)

        self.optimizer.zero_grad()

        # Compute features of base encoder
        base_features_1 = self.base_encoder(view_1)
        base_features_2 = self.base_encoder(view_2)

        # Compute projections of base projector
        base_projections_1 = self.base_projector(base_features_1)
        base_projections_2 = self.base_projector(base_features_2)

        # Compute predictions
        predictions_1 = self.predictor(base_projections_1)
        predictions_2 = self.predictor(base_projections_2)

        with torch.no_grad():  # disable gradient calculation

            # Compute features of momentum encoder
            momentum_features_1 = self.momentum_encoder(view_1)
            momentum_features_2 = self.momentum_encoder(view_2)

            # Compute features of momentum projector
            momentum_projections_1 = self.momentum_projector(momentum_features_1)
            momentum_projections_2 = self.momentum_projector(momentum_features_2)

        # Calculate loss
        batch_loss_list = list()

        loss = self.losses_fns[0](predictions_1, momentum_projections_2)\
            + self.losses_fns[0](predictions_2, momentum_projections_1)
        batch_loss_list.append(loss.detach().cpu().numpy())

        # Calculate metrics
        batch_metric_list = list()

        for metric_fn in self.metrics_fns:
            metric = metric_fn(predictions_1, momentum_projections_2)\
                + metric_fn(predictions_2, momentum_projections_1)
            batch_metric_list.append(metric.detach().cpu().numpy())

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return batch_loss_list, batch_metric_list

    def validation_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        self.base_encoder.eval()
        self.base_projector.eval()
        self.predictor.eval()

        view_1 = data[0].to(self.device)
        view_2 = data[1].to(self.device)

        with torch.no_grad():
            # Compute features of base encoder
            base_features_1 = self.base_encoder(view_1)
            base_features_2 = self.base_encoder(view_2)

            # Compute projections of base projector
            base_projections_1 = self.base_projector(base_features_1)
            base_projections_2 = self.base_projector(base_features_2)

            # Compute predictions
            predictions_1 = self.predictor(base_projections_1)
            predictions_2 = self.predictor(base_projections_2)

            # Compute features of momentum encoder
            momentum_features_1 = self.momentum_encoder(view_1)
            momentum_features_2 = self.momentum_encoder(view_2)

            # Compute features of momentum projector
            momentum_projections_1 = self.momentum_projector(momentum_features_1)
            momentum_projections_2 = self.momentum_projector(momentum_features_2)

            # Calculate loss
            batch_loss_list = list()

            loss = self.losses_fns[0](predictions_1, momentum_projections_2) \
                + self.losses_fns[0](predictions_2, momentum_projections_1)
            batch_loss_list.append(loss.detach().cpu().numpy())

            # Calculate metrics
            batch_metric_list = list()

            for metric_fn in self.metrics_fns:
                metric = metric_fn(predictions_1, momentum_projections_2) \
                    + metric_fn(predictions_2, momentum_projections_1)
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

        logger.info("Base projector architecture:")
        print(self.base_projector)

        logger.info("Momentum encoder architecture:")
        print(self.momentum_encoder)

        logger.info("Momentum projector architecture:")
        print(self.momentum_projector)

        logger.info("Predictor architecture:")
        print(self.predictor)
