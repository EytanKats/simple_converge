import tensorflow as tf
import efficientnet.tfkeras as efn
from tf_models.BaseModel import BaseModel


class EfficientNetB5(BaseModel):
    
    """
    This class encapsulates EfficientNetB5 model
    """

    def __init__(self):
        
        """
        This method initializes parameters
        :return: None 
        """

        super(EfficientNetB5, self).__init__()

        self.tensorflow_weights = None
        self.classes_num = 1000
        self.input_shape = (224, 224, 3)
        self.pooling = "avg"
        self.freeze_backbone_layers = False

        self.skip_connections = {
            2: "block2a_expand_activation",
            4: "block3a_expand_activation",
            8: "block4a_expand_activation",
            16: "block6a_expand_activation"
        }

    def parse_args(self, **kwargs):
        
        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(EfficientNetB5, self).parse_args(**kwargs)

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
        This method returns EfficientNetB5 model without top as defined in efficientnet package
        :return: EfficientNetB5 model without top as defined in efficientnet package
        """

        backbone = efn.EfficientNetB5(include_top=False,
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
        This method builds architecture of EfficientNetB5 model including top as defined in efficientnet package
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

            model = efn.EfficientNetB5(weights=self.tensorflow_weights,
                                       classes=self.classes_num)

        self.model = model
