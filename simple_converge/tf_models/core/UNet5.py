import tensorflow as tf
from simple_converge.tf_models.core.Conv2DBnX2 import Conv2DBnX2


class UNet5(tf.keras.Model):

    """
    This class implements UNet architecture that consists of 5 levels, utilizes max pooling in encoder part
    and up sampling in decoder part and.
    Convolution blocks have 3x3 kernels.
    Default number of filters:
    - in encoder blocks: 32, 64, 128, 256, 512.
    - in bottleneck block: 1024.
    - in decoder blocks: 512, 256, 128, 64, 32.
    Default number of classes: 2.
    Default output activation: "softmax"
    """

    def __init__(self,
                 num_filters=(32, 64, 128, 256, 512, 1024),
                 num_classes=2,
                 output_activation="softmax"):

        super(UNet5, self).__init__()

        # Fill model configuration parameters
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.output_activation = output_activation

        # Instantiate model layers
        self.encoder_conv_1 = Conv2DBnX2(filter_num=num_filters[0],
                                         kernel_size=(3, 3))

        self.max_pool_1 = tf.keras.layers.MaxPool2D(strides=2)

        self.encoder_conv_2 = Conv2DBnX2(filter_num=num_filters[1],
                                         kernel_size=(3, 3))

        self.max_pool_2 = tf.keras.layers.MaxPool2D(strides=2)

        self.encoder_conv_3 = Conv2DBnX2(filter_num=num_filters[2],
                                         kernel_size=(3, 3))

        self.max_pool_3 = tf.keras.layers.MaxPool2D(strides=2)

        self.encoder_conv_4 = Conv2DBnX2(filter_num=num_filters[3],
                                         kernel_size=(3, 3))

        self.max_pool_4 = tf.keras.layers.MaxPool2D(strides=2)

        self.encoder_conv_5 = Conv2DBnX2(filter_num=num_filters[4],
                                         kernel_size=(3, 3))

        self.max_pool_5 = tf.keras.layers.MaxPool2D(strides=2)

        self.bottleneck_conv = Conv2DBnX2(filter_num=num_filters[5],
                                          kernel_size=(3, 3))

        self.up_sampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.decoder_conv_1 = Conv2DBnX2(filter_num=num_filters[4],
                                         kernel_size=(3, 3))

        self.up_sampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.decoder_conv_2 = Conv2DBnX2(filter_num=num_filters[3],
                                         kernel_size=(3, 3))

        self.up_sampling_3 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.decoder_conv_3 = Conv2DBnX2(filter_num=num_filters[2],
                                         kernel_size=(3, 3))

        self.up_sampling_4 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.decoder_conv_4 = Conv2DBnX2(filter_num=num_filters[1],
                                         kernel_size=(3, 3))

        self.up_sampling_5 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.decoder_conv_5 = Conv2DBnX2(filter_num=num_filters[0],
                                         kernel_size=(3, 3))

        self.head_conv = tf.keras.layers.Conv2D(filters=num_classes,
                                                kernel_size=(1, 1))

        self.final_activation = tf.keras.layers.Activation(activation=output_activation)

    def get_config(self):

        model_configuration = {"num_filters": self.num_filters,
                               "num_classes": self.num_classes,
                               "output_activation": self.output_activation}

        return model_configuration

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def call(self,
             inputs,
             training=False,
             mask=None):

        skip_connection_1 = self.encoder_conv_1(inputs, training=training)
        x = self.max_pool_1(skip_connection_1)

        skip_connection_2 = self.encoder_conv_2(x, training=training)
        x = self.max_pool_2(skip_connection_2)

        skip_connection_3 = self.encoder_conv_3(x, training=training)
        x = self.max_pool_3(skip_connection_3)

        skip_connection_4 = self.encoder_conv_4(x, training=training)
        x = self.max_pool_4(skip_connection_4)

        skip_connection_5 = self.encoder_conv_5(x, training=training)
        x = self.max_pool_5(skip_connection_5)

        x = self.bottleneck_conv(x, training=training)

        x = self.up_sampling_1(x)
        x = tf.keras.layers.concatenate([x, skip_connection_5])
        x = self.decoder_conv_1(x, training=training)

        x = self.up_sampling_2(x)
        x = tf.keras.layers.concatenate([x, skip_connection_4])
        x = self.decoder_conv_2(x, training=training)

        x = self.up_sampling_3(x)
        x = tf.keras.layers.concatenate([x, skip_connection_3])
        x = self.decoder_conv_3(x, training=training)

        x = self.up_sampling_4(x)
        x = tf.keras.layers.concatenate([x, skip_connection_2])
        x = self.decoder_conv_4(x, training=training)

        x = self.up_sampling_5(x)
        x = tf.keras.layers.concatenate([x, skip_connection_1])
        x = self.decoder_conv_5(x, training=training)

        x = self.head_conv(x)
        output = self.final_activation(x)

        return output