import tensorflow as tf


class Conv2DBn(tf.keras.layers.Layer):

    """
    This class implements building block of CNN that consist of:
    - 2D convolutional layer with 'same' padding
    - batch normalization layer
    """

    def __init__(self,
                 filter_num,
                 kernel_size,
                 strides=(1, 1)):

        super(Conv2DBn, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filter_num,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="same")

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self,
             inputs,
             training=False):

        x = self.conv(inputs)
        output = self.bn(x, training=training)

        return output
