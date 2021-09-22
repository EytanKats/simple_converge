import tensorflow as tf

from classification_models.tfkeras import Classifiers
from simple_converge.tf_models.BaseModel import BaseModel


class Classifier (BaseModel):

    """
    This class encapsulates classifier implemented in classification_models package:
    https://github.com/qubvel/classification_models.
    Classifier can be created with different architectures all of which have pretrained on imagenet weights
    (default is resnet50):
    - vgg16, vgg19
    - resnet18, resnet34, resnet50, resnet101, resnet152
    - seresnet18, seresnet34, seresnet50, seresnet101, seresnet152
    - resnext50, resnext101
    - seresnext50, seresnext101
    - senet154
    - densenet121, densenet169, densenet201
    - inceptionv3, inceptionresnetv2, xception
    - mobilenet, mobilenetv2
    - nasnetlarge, nasnetmobile
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(Classifier, self).__init__()

        self.architecture = "resnet50"
        self.num_classes = 1000
        self.input_shape = (None, None, 3)
        self.include_top = False
        self.output_activation = "softmax"  # name of one of 'keras.activations' for last model layer
        self.weights = "imagenet"  # one of 'None' (random initialization), 'imagenet' (pre-training on ImageNet)
        self.freeze = False  # if 'True' set all layers (not including top) of 3rd party model as non-trainable

    def parse_args(self, **kwargs):

        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(Classifier, self).parse_args(**kwargs)

        if "architecture" in self.params.keys():
            self.architecture = self.params["architecture"]

        if "num_classes" in self.params.keys():
            self.num_classes = self.params["num_classes"]

        if "input_shape" in self.params.keys():
            self.input_shape = self.params["input_shape"]

        if "include_top" in self.params.keys():
            self.include_top = self.params["include_top"]

        if "output_activation" in self.params.keys():
            self.output_activation = self.params["output_activation"]

        if "weights" in self.params.keys():
            self.weights = self.params["weights"]

        if "freeze" in self.params.keys():
            self.freeze = self.params["freeze"]

    def build(self):

        """
        This method instantiates classifier model according to parameters
        :return: None
        """

        model_fn, _ = Classifiers.get(self.architecture)
        model = model_fn(input_shape=self.input_shape,
                         weights=self.weights,
                         classes=self.num_classes,
                         include_top=self.include_top)

        if not self.include_top:

            if self.freeze:
                for layer in model.layers:
                    layer.trainable = False

            x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
            output = tf.keras.layers.Dense(self.num_classes, activation=self.output_activation)(x)
            self.model = tf.keras.models.Model(inputs=[model.input], outputs=[output])

        else:
            self.model = model

