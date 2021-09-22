from simple_converge.tf_models.BaseModel import BaseModel
from simple_converge.tf_models.core.UNet5 import UNet5


class UNet(BaseModel):
    """
    This class encapsulates UNet models with different number of levels:
    - UNet5
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(UNet, self).__init__()

        self.unet_type = "unet5"
        self.num_classes = 2
        self.num_filters = (32, 64, 128, 256, 512, 1024)
        self.output_activation = "softmax"
        self.input_shape = (256, 256, 3)

        self.available_models = {"unet5": UNet5}

    def parse_args(self, **kwargs):

        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(UNet, self).parse_args(**kwargs)

        if "unet_type" in self.params.keys():
            self.unet_type = self.params["unet_type"]

        if "num_classes" in self.params.keys():
            self.num_classes = self.params["num_classes"]

        if "num_filters" in self.params.keys():
            self.num_filters = self.params["num_filters"]

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

        if self.unet_type in self.available_models.keys():
            self.model = self.available_models[self.unet_type](num_filters=self.num_filters,
                                                               num_classes=self.num_classes,
                                                               output_activation=self.output_activation)
        else:
            self.logger.log("Unknown type of model: {0}".format(self.unet_type))

        # Workaround to initialize model properly
        if self.model is not None:
            batch_input_shape = (None, *self.input_shape)
            self.model.build(input_shape=batch_input_shape)
            self.model.compute_output_shape(input_shape=batch_input_shape)