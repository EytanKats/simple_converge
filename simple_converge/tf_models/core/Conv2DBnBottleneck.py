import tensorflow as tf
from simple_converge.tf_models.core.Conv2DBn import Conv2DBn


class Conv2DBnBottleneck(tf.keras.layers.Layer):

    """
    This class implements building block of CNN that consist of:
    - Conv2DBn block with kernel size 1x1 and ReLU activation
    - Conv2DBn block with same padding and ReLU activation
    - Conv2DBn block with kernel size 1x1 and optional ReLU activation
    """

    def __init__(self,
                 filter_num,
                 kernel_size,
                 strides=(1, 1),
                 output_activation=True):

        super(Conv2DBnBottleneck, self).__init__()

        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_activation = output_activation

        self.conv_1 = Conv2DBn(filter_num=filter_num[0],
                               kernel_size=(1, 1),
                               output_activation=True)

        self.conv_2 = Conv2DBn(filter_num=filter_num[1],
                               kernel_size=kernel_size,
                               strides=strides,
                               output_activation=True)

        self.conv_3 = Conv2DBn(filter_num=filter_num[2],
                               kernel_size=(1, 1),
                               output_activation=output_activation)

    def get_config(self):

        config = super(Conv2DBnBottleneck, self).get_config()
        config.update({"filter_num": self.filter_num,
                       "kernel_size": self.kernel_size,
                       "strides": self.strides,
                       "output_activation": self.output_activation})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self,
             inputs,
             training=False):

        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)
        output = self.conv_3(x, training=training)

        return output
