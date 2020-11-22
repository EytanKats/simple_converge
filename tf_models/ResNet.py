import tensorflow as tf
from tf_models.BaseModel import BaseModel
from tf_models.core.ResNet18 import ResNet18


class ResNet(BaseModel):
    
    """
    This class encapsulates ResNet models:
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

    def build(self):

        """
        This method instantiate ResNet model according to 'resnet_type'
        :return: None
        """

        if self.resnet_type in self.available_models.keys():
            self.model = self.available_models[self.resnet_type](self.num_classes)
        else:
            self.logger.log("Unknown type of model: {0}".format(self.resnet_type))

