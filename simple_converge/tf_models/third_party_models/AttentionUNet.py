from keras_unet_collection import models
from simple_converge.tf_models.BaseModel import BaseModel


class AttentionUNet(BaseModel):
    
    """
    This class encapsulates AttentionUNet model implemented in keras_unet_collection package:
    https://github.com/yingkaisha/keras-unet-collection.
    UNet model can be created with different backbones all of which have pretrained on imagenet weights
    (default is ResNet50):
    - VGG16, VGG19
    - ResNet50, ResNet101, ResNet152
    - DenseNet121, DenseNet169, DenseNet201
    - EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5,
      EfficientNetB6, EfficientNetB7
    """

    def __init__(self):
        
        """
        This method initializes parameters
        :return: None 
        """

        super(AttentionUNet, self).__init__()

        self.input_shape = (None, None, 3)
        self.filters = (32, 64, 128, 256, 512)
        self.num_classes = 1
        self.attention_activation = "ReLU"
        self.output_activation = "Sigmoid"
        self.attention_type = "add"
        self.use_batch_normalization = True
        self.pooling = "max"
        self.unpooling = "bilinear"
        self.backbone = "ResNet50"
        self.encoder_weights = "imagenet"
        self.encoder_freeze = False

    def parse_args(self, **kwargs):
        
        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(AttentionUNet, self).parse_args(**kwargs)

        if "input_shape" in self.params.keys():
            self.input_shape = self.params["input_shape"]

        if "filters" in self.params.keys():
            self.filters = self.params["filters"]

        if "num_classes" in self.params.keys():
            self.num_classes = self.params["num_classes"]

        if "attention_activation" in self.params.keys():
            self.attention_activation = self.params["attention_activation"]

        if "output_activation" in self.params.keys():
            self.output_activation = self.params["output_activation"]

        if "attention_type" in self.params.keys():
            self.attention_type = self.params["attention_type"]

        if "use_batch_normalization" in self.params.keys():
            self.use_batch_normalization = self.params["use_batch_normalization"]

        if "pooling" in self.params.keys():
            self.pooling = self.params["pooling"]

        if "unpooling" in self.params.keys():
            self.unpooling = self.params["unpooling"]

        if "backbone" in self.params.keys():
            self.backbone = self.params["backbone"]

        if "encoder_weights" in self.params.keys():
            self.encoder_weights = self.params["encoder_weights"]

        if "encoder_freeze" in self.params.keys():
            self.encoder_freeze = self.params["encoder_freeze"]

    def build(self):

        """
        This method instantiates AttentionUNet model according to parameters
        :return: None
        """

        self.model = models.att_unet_2d(input_size=self.input_shape,
                                        filter_num=self.filters,
                                        n_labels=self.num_classes,
                                        atten_activation=self.attention_activation,
                                        output_activation=self.output_activation,
                                        attention=self.attention_type,
                                        batch_norm=self.use_batch_normalization,
                                        pool=self.pooling,
                                        unpool=self.unpooling,
                                        backbone=self.backbone,
                                        weights=self.encoder_weights,
                                        freeze_backbone=self.encoder_freeze)
