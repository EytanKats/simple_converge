import tensorflow as tf
from simple_converge.tf_models.core.Conv2DBn import Conv2DBn


class Conv2DBnX2(tf.keras.layers.Layer):

    """
    This class implements block which consist of 2 concatenated Conv2DBn blocks.
    Both blocks have (1, 1) strides and ReLU activation.
    """

    def __init__(self,
                 filter_num,
                 kernel_size):

        super(Conv2DBnX2, self).__init__()

        self.filter_num = filter_num
        self.kernel_size = kernel_size

        self.conv_1 = Conv2DBn(filter_num=filter_num,
                               kernel_size=kernel_size,
                               strides=(1, 1),
                               output_activation=True)

        self.conv_2 = Conv2DBn(filter_num=filter_num,
                               kernel_size=kernel_size,
                               strides=(1, 1),
                               output_activation=True)

    def get_config(self):

        config = super(Conv2DBnX2, self).get_config()
        config.update({"filter_num": self.filter_num,
                       "kernel_size": self.kernel_size})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self,
             inputs,
             training=False):

        x = self.conv_1(inputs, training=training)
        output = self.conv_2(x, training=training)

        return output