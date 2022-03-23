import torch
from loguru import logger
from apps.BaseSingleModelApp import BaseSingleModelApp

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


class SingleModelApp(BaseSingleModelApp):

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
        
        super(SingleModelApp, self).__init__(
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
        )

        self.ckpt_cnt = 0
        self.latest_ckpt_path = None

    def _get_latest_ckpt(self):
        return self.latest_ckpt_path

    def _restore_ckpt(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))

    def _save_ckpt(self, ckpt_path):
        self.latest_ckpt_path = ckpt_path + '-' + str(self.ckpt_cnt) + '.pth'
        self.ckpt_cnt += 1
        torch.save(self.model.state_dict(), self.latest_ckpt_path)

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _step(self, data, labels, training):

        if training:
            self.model.train()
        else:
            self.model.eval()

        data = data.cuda()
        labels = labels.cuda()

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            # Forward pass
            model_output = self.model(data)

            # Calculate loss
            batch_loss = 0
            batch_loss_list = list()

            if len(self.loss_fns) == 1:
                loss = self.loss_fns[0](model_output, labels)
                batch_loss_list.append(loss.detach().cpu().numpy())
                batch_loss += loss * self.loss_weights[0]
            else:
                for int_loss_idx in range(len(self.loss_fns)):
                    loss = self.loss_fns[int_loss_idx](model_output[int_loss_idx], labels[int_loss_idx])
                    batch_loss_list.append(loss.detach().cpu().numpy())
                    batch_loss += loss * self.loss_weights[int_loss_idx]

            # Calculate metrics
            batch_metric_list = list()

            if len(self.metric_fns) == 1:
                for metric_fn in self.metric_fns[0]:
                    metric = metric_fn(model_output, labels)
                    batch_metric_list.append(metric.detach().cpu().numpy())

            else:
                for int_output_idx, metric_fns_for_output in enumerate(self.metric_fns):
                    for metric_fn in metric_fns_for_output:
                        metric = metric_fn(model_output[int_output_idx], labels[int_output_idx])
                        batch_metric_list.append(metric.detach().cpu().numpy())

        # Backward pass
        if training:
            loss.backward()
            self.optimizer.step()

        return batch_loss_list, batch_metric_list

    def predict(self, data):

        self.model.eval()
        data = data.cuda()
        with torch.no_grad():
            predictions = self.model(data)
        predictions = predictions.detach().cpu().numpy()
        return predictions

    def summary(self):
        logger.info("Model architecture:")
        print(self.model)
