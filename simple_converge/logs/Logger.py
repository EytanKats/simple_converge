import os
import logging

from simple_converge.logs.LogLevels import LogLevels
from simple_converge.base.BaseObject import BaseObject


class Logger(BaseObject):

    """
    This class encapsulates build-in python logger
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(Logger, self).__init__()

        self.file_name = "log.txt"
        self.message_format = "%(message)s"

        self.file_path = ""
        self.file_handler = None

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Logger, self).parse_args(**kwargs)

        if "file_name" in self.params.keys():
            self.file_name = self.params["file_name"]

        if "message_format" in self.params.keys():
            self.message_format = self.params["message_format"]

    def start(self, folder):

        if self.logger is not None:

            print("This logger is already started. View logs in {0}".format(self.file_path))
            return

        self.logger = logging.getLogger('FileLogger')
        self.logger.setLevel(logging.DEBUG)

        self.file_path = os.path.join(folder, self.file_name)
        self.file_handler = logging.FileHandler(self.file_path, mode='w')
        self.file_handler.setFormatter(logging.Formatter(self.message_format))
        self.logger.addHandler(self.file_handler)

    def end(self):
        self.logger.removeHandler(self.file_handler)
        self.file_handler = None

    def log(self, message, level=LogLevels.DEBUG, console=True):
        if self.file_handler is None:
            print("File handler is not defined, call start method")

        if console:
            print(message)

        if level == LogLevels.DEBUG:
            self.logger.debug(message)
        elif level == LogLevels.INFO:
            self.logger.info(message)
        elif level == LogLevels.WARNING:
            self.logger.warning(message)
        elif level == LogLevels.ERROR:
            self.logger.error(message)
        elif level == LogLevels.CRITICAL:
            self.logger.critical(message)
