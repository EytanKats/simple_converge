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

        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_activation = output_activation

        self.conv = tf.keras.layers.Conv2D(filters=filter_num,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="same")

        self.bn = tf.keras.layers.BatchNormalization()

    def get_config(self):

        config = super(Conv2DBn, self).get_config()
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

        x = self.conv(inputs)
        x = self.bn(x, training=training)

        if self.output_activation:
            output = tf.nn.relu(x)
        else:
            output = x

        return output
