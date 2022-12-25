import torch
import numpy as np

from simple_converge.utils.training import EMA, BnController
from simple_converge.apps.SingleModelApp import SingleModelApp


default_settings = {
    'use_reduce_lr_on_plateau': False,
    'reduce_lr_on_plateau_factor': 0.8,
    'reduce_lr_on_plateau_min': 1e-6,
    'max_consistency_loss_weight': 0,
    'consistency_ramp_up': 0,
    'no_label_idx': 0,
    'use_ema': False,
    'ema_decay': 0.999,
    'use_bn_controller': False
}


class PiModelApp(SingleModelApp):

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
            optimizer
            ):
        
        """
        This method initializes parameters
        :return: None 
        """
        
        super(PiModelApp, self).__init__(
            settings,
            mlops_task,
            architecture,
            loss_function,
            metric,
            scheduler,
            optimizer
        )

        self.bn_controller = BnController()

    def training_step(self, data, epoch):

        """
        Training step for Pi Model application.
        :param data: Tuple that contains to different views of same data samples (same data samples that are augmented randomly).
        :param labels:
        :return:
        """

        self.model.train()

        # Send data and labels to GPU
        sup_input_data = data[0].to(self.device)
        sup_labels = data[1].to(self.device)
        unsup_data_view_1 = data[3].to(self.device)
        unsup_data_view_2 = data[4].to(self.device)

        # Reset optimizer
        self.optimizer.zero_grad()

        # Calculate consistency weight
        if self.settings['app']['consistency_ramp_up'] == 0:
            consistency_loss_weight = self.settings['app']['max_consistency_loss_weight']
        else:
            current = np.clip(epoch, 0.0, self.settings['app']['consistency_ramp_up'])
            phase = 1.0 - current / self.settings['app']['consistency_ramp_up']
            consistency_loss_weight = self.settings['app']['max_consistency_loss_weight'] * float(np.exp(-5.0 * phase * phase))

        with torch.set_grad_enabled(True):

            # Forward pass
            sup_output = self.model(sup_input_data)

            if self.settings['app']['use_bn_controller']:
                self.bn_controller.freeze_bn(self.model)
            unsup_output_1 = self.model(unsup_data_view_1)
            unsup_output_2 = self.model(unsup_data_view_2)
            if self.settings['app']['use_bn_controller']:
                self.bn_controller.unfreeze_bn(self.model)

            # Calculate loss
            batch_loss_list = list()

            # Apply classification loss
            classification_loss = self.losses_fns[0](sup_output, sup_labels)
            batch_loss_list.append(classification_loss.detach().cpu().numpy())

            # Apply consistency loss
            consistency_loss = self.losses_fns[1](unsup_output_1, unsup_output_2)
            batch_loss_list.append(consistency_loss.detach().cpu().numpy())

            # Calculate total loss
            loss = classification_loss + consistency_loss_weight * consistency_loss

            # Calculate metrics
            batch_metric_list = list()

            for metric_fn in self.metrics_fns:
                metric = metric_fn(sup_output, sup_labels)
                batch_metric_list.append(metric.detach().cpu().numpy())

        # Backward pass
        loss.backward()
        self.optimizer.step()

        if self.settings['app']['use_ema']:
            self.ema.update()

        return batch_loss_list, batch_metric_list

    def validation_step(self, data, epoch):

        """
        Validation step for Pi Model application.
        :param data:
        :param labels:
        :return:
        """

        self.model.eval()

        if self.settings['app']['use_ema']:
            self.ema.apply_shadow()

        # Send data and labels to GPU
        input_data = data[0].to(self.device)
        labels = data[1].to(self.device)

        with torch.set_grad_enabled(False):

            # Forward pass
            model_output = self.model(input_data)

            # Calculate loss
            batch_loss_list = list()

            classification_loss = self.losses_fns[0](model_output, labels)
            batch_loss_list.append(classification_loss.detach().cpu().numpy())

            # Add to list '0' as a placeholder for consistency loss
            batch_loss_list.append(0)

            # Calculate metrics
            batch_metric_list = list()

            for metric_fn in self.metrics_fns:
                metric = metric_fn(model_output, labels)
                batch_metric_list.append(metric.detach().cpu().numpy())

        if self.settings['app']['use_ema']:
            self.ema.restore()

        return batch_loss_list, batch_metric_list
