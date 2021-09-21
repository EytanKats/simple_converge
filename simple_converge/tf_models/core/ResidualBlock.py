import tensorflow as tf
from simple_converge.tf_models.core.Conv2DBn import Conv2DBn


class ResidualBlock(tf.keras.layers.Layer):

    """
    This class implements residual block which consist of 2 Conv2DBn blocks and shortcut connection.
    Strides are applied in first Conv2DBn block, second Conv2DBn block always have strides (1, 1).
    If strides different from (1, 1) than shortcut connection includes Conv2DBn block with kernel size (1, 1)
    and specified strides (projection shortcut), else shortcut connection is identity.
    """

    def __init__(self,
                 filter_num,
                 kernel_size,
                 strides=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv_1 = Conv2DBn(filter_num=filter_num,
                               kernel_size=kernel_size,
                               strides=strides,
                               output_activation=True)

        self.conv_2 = Conv2DBn(filter_num=filter_num,
                               kernel_size=kernel_size,
                               strides=(1, 1),
                               output_activation=False)

        self.strides = strides
        if strides[0] != 1 or strides[1] != 1:
            self.shortcut = Conv2DBn(filter_num=filter_num,
                                     kernel_size=(1, 1),
                                     strides=strides,
                                     output_activation=False)

    def get_config(self):

        config = super(ResidualBlock, self).get_config()
        config.update({"filter_num": self.filter_num,
                       "kernel_size": self.kernel_size,
                       "strides": self.strides})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self,
             inputs,
             training=False):

        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)

        if self.strides[0] != 1 or self.strides[1] != 1:
            residual = self.shortcut(inputs, training=training)
        else:
            residual = inputs

        x = tf.keras.layers.add([residual, x])
        output = tf.nn.relu(x)

        return output
