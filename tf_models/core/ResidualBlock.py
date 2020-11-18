import tensorflow as tf
from tf_models.core.Conv2DBn import Conv2DBn


class ResidualBlock(tf.keras.layers.Layer):

    """
    This class implements residual block which consist of 2 conv2d-bn-relu blocks and shortcut connection.
    Residual block supports both identity and projection shortcut.
    """

    def __init__(self,
                 filter_num,
                 kernel_size,
                 strides=(1, 1)):

        super(ResidualBlock, self).__init__()

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
