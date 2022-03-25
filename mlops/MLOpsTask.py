from clearml import Task


default_mlops_settings = {
    'project_name': 'Default Project',
    'task_name': 'default_task',
    'connect_frameworks': {
        'matplotlib': False,
        'tensorflow': False,
        'tensorboard': False,
        'pytorch': False,
        'xgboost': False,
        'scikit': False,
        'fastai': False,
        'lightgbm': False,
        'hydra': False
    }
}


class MLOpsTask(object):

    """
    This class instantiates encapsulates ClearML instance of MLOps task
    -
    """

    def __init__(
            self,
            settings
    ):

        """
        This method initializes parameters
        :return: None
        """

        self.settings = settings
        self._task = None

    @property
    def task(self):

        if not self.settings["use_mlops"]:
            return None

        if self._task is not None:
            return self._task

        self._task = Task.init(project_name=self.settings["project_name"],
                               task_name=self.settings["task_name"],
                               auto_connect_frameworks=self.settings["connect_frameworks"])

        return self._task

    def log_configuration(self, config_dict, name):
        self.task.connect(config_dict, name)
