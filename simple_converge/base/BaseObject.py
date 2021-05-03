import abc


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

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        self.params = kwargs["params"]

    def set_logger(self, logger):

        """
        This method sets logger to be used for this object
        :param logger: logger instance
        :return: None
        """

        self.logger = logger
