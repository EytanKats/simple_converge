from simple_converge.base.BaseObject import BaseObject


class BaseRegularizer(BaseObject):

    """
    This abstract class defines common methods to Tensorflow regularizers
    """

    def __init__(self, **kwargs):

        """
        This method initializes parameters
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseRegularizer, self).__init__()

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseRegularizer, self).parse_args(**kwargs)

    def get_regularizer(self):

        """
        This method returns regularizer
        :return: regularizer
        """

        pass
