from clearml import Task, TaskTypes

default_mlops_settings = {
    'project_name': 'Default Project',
    'task_name': 'default_task',
    'task_type': TaskTypes.training,
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

        self._task.set_task_type(self.settings['task_type'])

        return self._task

    def log_configuration(self, config_dict, name):
        if self.task:
            self.task.connect(config_dict, name)

    def log_scalar_to_mlops_server(self, plot_name, curve_name, val, iteration):
        if self.task:
            logger = self.task.get_logger()
            logger.report_scalar(plot_name, curve_name, val, iteration=iteration)

    def report_matplotlib_figure(self, figure, title, iteration):
        if self.task:
            mlops_logger = self.task.get_logger()
            mlops_logger.report_matplotlib_figure(
                title=title,
                series=title,
                iteration=iteration,
                figure=figure,
                report_image=True)
