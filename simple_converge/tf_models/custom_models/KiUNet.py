from simple_converge.tf_models.BaseModel import BaseModel
from simple_converge.tf_models.core.KiUNet3 import KiUNet3


class KiUNet(BaseModel):
    """
    This class encapsulates KiUNet models with different number of levels:
    - KiUNet3
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(KiUNet, self).__init__()

        self.kiu_net_type = "kiu_net_3"
        self.num_encoder_filters = (16, 32, 64)
        self.num_decoder_filters = (32, 16, 8)
        self.num_classes = 2
        self.output_activation = "softmax"
        self.input_shape = (256, 256, 3)

        self.available_models = {"kiu_net_3": KiUNet3}

    def parse_args(self, **kwargs):

        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(KiUNet, self).parse_args(**kwargs)

        if "kiu_net_type" in self.params.keys():
            self.kiu_net_type = self.params["kiu_net_type"]

        if "num_encoder_filters" in self.params.keys():
            self.num_encoder_filters = self.params["num_encoder_filters"]

        if "num_decoder_filters" in self.params.keys():
            self.num_decoder_filters = self.params["num_decoder_filters"]

        if "num_classes" in self.params.keys():
            self.num_classes = self.params["num_classes"]

        if "output_activation" in self.params.keys():
            self.output_activation = self.params["output_activation"]

        if "input_shape" in self.params.keys():
            self.input_shape = self.params["input_shape"]

    def build(self):

        """
        This method instantiates model according to its type
        and builds it to create weights.
        :return: None
        """

        if self.kiu_net_type in self.available_models.keys():
            self.model = self.available_models[self.kiu_net_type](num_encoder_filters=self.num_encoder_filters,
                                                                  num_decoder_filters=self.num_decoder_filters,
                                                                  num_classes=self.num_classes,
                                                                  output_activation=self.output_activation)
        else:
            self.logger.log("Unknown type of model: {0}".format(self.kiu_net_type))

        # Workaround to initialize model properly
        if self.model is not None:
            batch_input_shape = (None, *self.input_shape)
            self.model.build(input_shape=batch_input_shape)
            self.model.compute_output_shape(input_shape=batch_input_shape)
