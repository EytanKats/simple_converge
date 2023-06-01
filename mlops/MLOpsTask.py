from clearml import Task, TaskTypes

default_mlops_settings = {
    'project_name': 'Default Project',
    'task_name': 'default_task',
    'task_type': 'training',
    'tags': ['default_tag'],
    'connect_arg_parser': False,
    'connect_frameworks': False,
    'resource_monitoring': True,
    'connect_streams': True
}

task_type_map = {
    'training': TaskTypes.training,
    'testing': TaskTypes.testing
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

        task_type = task_type_map[self.settings['task_type']]
        self._task = Task.init(project_name=self.settings['project_name'],
                               task_name=self.settings['task_name'],
                               task_type=task_type,
                               tags=self.settings['tags'],
                               auto_connect_arg_parser=self.settings['connect_arg_parser'],
                               auto_connect_frameworks=self.settings['connect_frameworks'],
                               auto_resource_monitoring=self.settings['resource_monitoring'],
                               auto_connect_streams=self.settings['connect_streams'])

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
