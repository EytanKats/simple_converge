from simple_converge.base.BaseObject import BaseObject


class BaseOptimizer(BaseObject):

    """
    This abstract class defines common methods to Tensorflow optimizers
    """

    def __init__(self, **kwargs):

        """
        This method initializes parameters
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseOptimizer, self).__init__()

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseOptimizer, self).parse_args(**kwargs)

    def get_optimizer(self):

        """
        This method returns optimizer
        :return: optimizer
        """

        pass
