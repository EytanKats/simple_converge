import tensorflow as tf
from tf_models.BaseModel import BaseModel
from tf_models.backbones_collection import backbones_collection


class ClassificationNet(BaseModel):

    """
    This class defines simple classification to play with it
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(ClassificationNet, self).__init__()

        # Fields to be filled by parsing
        self.input_shape = (28, 28, 1)

    def parse_args(self, **kwargs):

        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(ClassificationNet, self).parse_args(**kwargs)

        params = kwargs["params"]

        if "input_shape" in self.params.keys():
            self.input_shape = self.params["input_shape"]

    def build(self):

        """
        This method builds architecture of the model
        :return: None
        """

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(units=512, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        outputs = tf.keras.layers.Dense(units=10, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
