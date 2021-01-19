import tensorflow as tf
from tf_models.BaseModel import BaseModel


class ResNet50V2(BaseModel):
    
    """
    This class encapsulates ResNet50 model
    """

    def __init__(self):
        
        """
        This method initializes parameters
        :return: None 
        """

        super(ResNet50V2, self).__init__()

        self.tensorflow_weights = None
        self.classes_num = 1000
        self.input_shape = (224, 224, 3)
        self.pooling = "avg"
        self.freeze_backbone_layers = False

        self.skip_connections = {
            2: "conv1_conv",
            4: "conv2_block2_out",
            8: "conv3_block3_out",
            16: "conv4_block5_out"
        }

    def parse_args(self, **kwargs):
        
        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(ResNet50V2, self).parse_args(**kwargs)

        if "tensorflow_weights" in self.params.keys():
            self.tensorflow_weights = self.params["tensorflow_weights"]

        if "classes_num" in self.params.keys():
            self.classes_num = self.params["classes_num"]

        if "input_shape" in self.params.keys():
            self.input_shape = self.params["input_shape"]

        if "pooling" in self.params.keys():
            self.pooling = self.params["pooling"]

        if "freeze_backbone_layers" in self.params.keys():
            self.freeze_backbone_layers = self.params["freeze_backbone_layers"]

    def get_backbone(self):

        """
        This method returns ResNet50V2 model without top as defined in tensorflow.keras
        :return: ResNet50V2 model without top as defined in tensorflow.keras
        """

        backbone = tf.keras.applications.ResNet50V2(include_top=False,
                                                    weights=self.tensorflow_weights,
                                                    input_shape=self.input_shape,
                                                    pooling=self.pooling)

        for layer in backbone.layers:
            layer.trainable = not self.freeze_backbone_layers

        return backbone

    def get_skip_connection(self, backbone, pooling_ratio):

        """
        This method returns output of skip connection layer for each pooling ratio
        :param backbone: model that contains pooling layers
        :param pooling_ratio: pooling ratio for which skip connections will be returned
        :return: output of skip connection layer
        """

        layer_name = self.skip_connections[pooling_ratio]
        layer = backbone.get_layer(name=layer_name)

        return layer.output

    def build(self):

        """
        This method builds architecture of ResNet50V2 model including top as defined in tensorflow.keras
        :return: None
        """

        if self.classes_num != 1000 and self.tensorflow_weights == "imagenet":  # in this case we need to add custom top

            backbone = self.get_backbone()
            x = backbone.output
            outputs = tf.keras.layers.Dense(units=self.classes_num,
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self._get_regularizer(),
                                            bias_regularizer=self._get_regularizer(),
                                            activation="softmax")(x)

            model = tf.keras.Model(inputs=backbone.input, outputs=outputs)

        else:

            model = tf.keras.applications.ResNet50V2(weights=self.tensorflow_weights,
                                                     classes=self.classes_num)

        self.model = model
