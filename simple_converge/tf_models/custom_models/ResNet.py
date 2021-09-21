from simple_converge.tf_models.BaseModel import BaseModel
from simple_converge.tf_models.core.ResNet18 import ResNet18


class ResNet(BaseModel):
    
    """
    This class encapsulates ResNet models of variable depths:
    - ResNet18
    """

    def __init__(self):
        
        """
        This method initializes parameters
        :return: None 
        """

        super(ResNet, self).__init__()

        self.resnet_type = "resnet18"
        self.num_classes = 1000
        self.input_shape = (256, 256, 3)

        self.available_models = {"resnet18": ResNet18}

    def parse_args(self, **kwargs):
        
        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(ResNet, self).parse_args(**kwargs)

        if "resnet_type" in self.params.keys():
            self.resnet_type = self.params["resnet_type"]

        if "num_classes" in self.params.keys():
            self.num_classes = self.params["num_classes"]

        if "input_shape" in self.params.keys():
            self.input_shape = self.params["input_shape"]

    def build(self):

        """
        This method instantiates model according to its type
        and builds it to create weights.
        :return: None
        """

        if self.resnet_type in self.available_models.keys():
            self.model = self.available_models[self.resnet_type](self.num_classes)
        else:
            self.logger.log("Unknown type of model: {0}".format(self.resnet_type))

        # Workaround to initialize model properly
        if self.model is not None:
            batch_input_shape = (None, *self.input_shape)
            self.model.build(input_shape=batch_input_shape)
            self.model.compute_output_shape(input_shape=batch_input_shape)
