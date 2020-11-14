import tensorflow as tf


class Conv2DBnRelu(tf.keras.layers.Layer):

    """
    This class implements building block of CNN that consist of:
    - 2D convolutional layer
    - batch normalization layer
    - ReLU activation
    """

    def __init__(self,
                 filter_num,
                 kernel_size):

        super(Conv2DBnRelu, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filter_num,
                                           kernel_size=kernel_size,
                                           padding="same")

        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.nn.relu()

    def call(self,
             inputs,
             training=False):

        x = self.conv(inputs)
        x = self.bn(x, training=training)
        output = tf.nn.relu(x)

        return output