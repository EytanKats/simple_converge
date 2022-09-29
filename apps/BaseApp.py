import abc
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger


default_settings = {
}


class BaseApp(abc.ABC):

    """
    This class defines interface that all applications have to implement.
    'Trainer' and 'Manager' uses defined interface to run application training.
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    @abc.abstractmethod
    def training_step(self, data, epoch):
        pass

    @abc.abstractmethod
    def validation_step(self, data, epoch):
        pass

    @abc.abstractmethod
    def predict(self, data):
        # return predictions
        pass

    @abc.abstractmethod
    def summary(self):
        pass
