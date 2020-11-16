import tensorflow as tf
from tf_models.core.Conv2DBn import Conv2DBn

class BottleneckConv2DBn(tf.keras.layers.Layer):

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

        super(BottleneckConv2DBn, self).__init__()

        self.conv_1 = Conv2DBn(filter_num=filter_num[0],
                               kernel_size=(1, 1),
                               output_activation=True)

        self.conv_2 = Conv2DBn(filter_num=filter_num[0],
                               kernel_size=kernel_size,
                               strides=strides,
                               output_activation=True)

        self.conv_3 = Conv2DBn(filter_num=filter_num[0],
                               kernel_size=(1, 1),
                               output_activation=output_activation)

    def call(self,
             inputs,
             training=False):

        x = self.conv_1(inputs)
        x = self.conv_2(x)
        output = self.conv_3(x)

        return output
