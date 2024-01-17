import wandb


default_mlops_settings = {
    'use_mlops': False,
    'project_name': 'Default Project',
    'task_name': 'default_task',
}


class MLOpsTask(object):

    """
    This class instantiates encapsulates wandb logic
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

        if self.settings['mlops']['use_mlops']:
            wandb.init(
                project=self.settings['mlops']['project_name'],
                name=self.settings['mlops']['task_name'],
                config=settings
            )

    def log_scalar_to_mlops_server(self, plot_name, curve_name, val, iteration):
        if self.settings['mlops']['use_mlops']:
            wandb.log({curve_name: val}, step=iteration)

    def report_matplotlib_figure(self, figure, title, iteration):
        if self.settings['mlops']['use_mlops']:
            wandb.log({title: figure}, step=iteration)
