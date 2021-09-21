import segmentation_models as sm
from simple_converge.tf_models.BaseModel import BaseModel


class UNet(BaseModel):
    
    """
    This class encapsulates UNet model implemented in segmentation_models package:
    https://github.com/qubvel/segmentation_models.
    UNet model can be created with different backbones all of which have pretrained on imagenet weights
    (default is resnet50):
    - vgg16, vgg19
    - resnet18, resnet34, resnet50, resnet101, resnet152
    - seresnet18, seresnet34, seresnet50, seresnet101, seresnet152
    - resnext50, resnext101
    - seresnext50, seresnext101
    - senet154
    - densenet121, densenet169, densenet201
    - inceptionv3, inceptionresnetv2
    - mobilenet, mobilenetv2
    - efficientnetb0, efficientnetb1, efficientnetb2, efficientnetb3, efficientnetb4, efficientnetb5,
      efficientnetb6, efficientnetb7
    """

    def __init__(self):
        
        """
        This method initializes parameters
        :return: None 
        """

        super(UNet, self).__init__()

        self.backbone = "resnet50"
        self.num_classes = 1
        self.input_shape = (None, None, 3)
        self.output_activation = "sigmoid"  # name of one of 'keras.activations' for last model layer
        self.encoder_weights = "imagenet"  # one of 'None' (random initialization), 'imagenet' (pre-training on ImageNet)
        self.encoder_freeze = False  # if 'True' set all layers of encoder (backbone model) as non-trainable.
        self.decoder_block_type = "upsampling"  # one of "upsampling" or "transpose"
        self.decoder_filters = (256, 128, 64, 32, 16)
        self.decoder_use_batchnorm = True

    def parse_args(self, **kwargs):
        
        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(UNet, self).parse_args(**kwargs)

        if "backbone" in self.params.keys():
            self.backbone = self.params["backbone"]

        if "num_classes" in self.params.keys():
            self.num_classes = self.params["num_classes"]

        if "input_shape" in self.params.keys():
            self.input_shape = self.params["input_shape"]

        if "output_activation" in self.params.keys():
            self.output_activation = self.params["output_activation"]

        if "encoder_weights" in self.params.keys():
            self.encoder_weights = self.params["encoder_weights"]

        if "encoder_freeze" in self.params.keys():
            self.encoder_freeze = self.params["encoder_freeze"]

        if "decoder_block_type" in self.params.keys():
            self.decoder_block_type = self.params["decoder_block_type"]

        if "decoder_filters" in self.params.keys():
            self.decoder_filters = self.params["decoder_filters"]

        if "decoder_use_batchnorm" in self.params.keys():
            self.decoder_use_batchnorm = self.params["decoder_use_batchnorm"]

    def build(self):

        """
        This method instantiates UNet model according to parameters
        :return: None
        """

        self.model = sm.Unet(backbone_name=self.backbone,
                             input_shape=self.input_shape,
                             classes=self.num_classes,
                             activation=self.output_activation,
                             encoder_weights=self.encoder_weights,
                             encoder_freeze=self.encoder_freeze,
                             decoder_block_type=self.decoder_block_type,
                             decoder_filters=self.decoder_filters,
                             decoder_use_batchnorm=self.decoder_use_batchnorm)
