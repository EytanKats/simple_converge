from simple_converge.base.BaseObject import BaseObject


class BaseMetric(BaseObject):
    """
    This abstract class defines common methods for Tensorflow metrics
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(BaseMetric, self).__init__()

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(BaseMetric, self).parse_args(**kwargs)
        
    def get_metric(self):

        """
        This method returns loss function
        :return: loss function
        """

        pass
