import abc
import torch
import simple_converge as sc


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
            mlops_task,
            loss_function,
            metric
    ):
        
        """
        This method initializes parameters.
        :param settings: Dictionary that contains configuration parameters.
        :return: None 
        """
        
        self.settings = settings
        self.mlops_task = mlops_task

        if loss_function is not None:
            self.losses_fns = loss_function(settings)
            self.losses_names = [_loss.__name__ for _loss in self.losses_fns]
            self.losses_num = len(self.losses_names)
        elif 'registry_name' in settings['loss']:
            self.losses_fns = [sc.loss_functions.Registry[loss_name](settings)
                               for loss_name in settings['loss']['registry_name']]
            self.losses_names = [_loss.__name__ for _loss in self.losses_fns]
            self.losses_num = len(self.losses_names)
        else:
            self.losses_fns = None

        if metric is not None:
            self.metrics_fns = metric(settings)
            self.metrics_names = [_metric.__name__ for _metric in self.metrics_fns]
            self.metrics_num = len(self.metrics_names)
        else:
            self.metrics_fns = None

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
    def on_epoch_start(self):
        pass

    @abc.abstractmethod
    def on_epoch_end(self, is_plateau=False):
        pass

    @abc.abstractmethod
    def training_step(self, data, epoch, cur_iteration, iterations_per_epoch):
        pass

    @abc.abstractmethod
    def validation_step(self, data, epoch, cur_iteration, iterations_per_epoch):
        pass

    @abc.abstractmethod
    def predict(self, data):
        # return predictions
        pass

    @abc.abstractmethod
    def summary(self):
        pass
