import tensorflow as tf


class Conv2DBn(tf.keras.layers.Layer):

    """
    This class implements building block of CNN that consist of:
    - 2D convolutional layer with 'same' padding
    - batch normalization layer
    - ReLU activation (optional)
    """

    def __init__(self,
                 filter_num,
                 kernel_size,
                 strides=(1, 1),
                 output_activation=True):

        super(Conv2DBn, self).__init__()

        self.output_activation = output_activation

        self.conv = tf.keras.layers.Conv2D(filters=filter_num,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="same")

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self,
             inputs,
             training=False):

        x = self.conv(inputs)
        x = self.bn(x, training=training)

        if self.output_activation:
            output = tf.nn.relu(x)
        else:
            output = x

        return output
