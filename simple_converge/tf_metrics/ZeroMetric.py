from simple_converge.tf_metrics.BaseMetric import BaseMetric


class ZeroMetric(BaseMetric):

    """
    This class implements the zero metric.
    The zero metric is useful when there is no need to apply loss on the specific output of the model.
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(ZeroMetric, self).__init__()

    def parse_args(self, **kwargs):

        """
        This method sets values of parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(ZeroMetric, self).parse_args(**kwargs)

    def get_metric(self):

        """
        This method returns metric function according to parameters
        :return: metric function
        """

        return self.zero_metric

    @staticmethod
    def zero_metric(y_true, y_pred):

        """
        This method returns zero to not propagate any gradients
        :return: 0
        """

        return 0.
