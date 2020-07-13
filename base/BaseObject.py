import abc
from logger.LogLevels import LogLevels


class BaseObject(abc.ABC):
    """
    This abstract class defines common methods to all classes in this framework
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        self.params = dict()
        self.logger = None

    def _log(self, message, level=LogLevels.DEBUG, console=True):

        """
        This method log the message with defined logger.
        If logger is not available the message will be printed to console.
        :param message: message to log
        :param level: severity level of log message
        :param console: if true message will be printed also to console
        :return: None
        """

        if self.logger is not None:
            self.logger.log(message, level, console)
        else:
            print(message)

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        self.params = kwargs["params"]

        if "logger" in self.params.keys():
            self.logger = self.params["logger"]
