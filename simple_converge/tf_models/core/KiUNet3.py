import tensorflow as tf
from simple_converge.tf_models.core.Conv2DBn import Conv2DBn


class KiUNet3(tf.keras.Model):

    """
    This class implements KiUNet architecture that consists of 3 levels and specified in following paper: https://arxiv.org/abs/2006.04878
    Convolution blocks have 3x3 kernels.
    Default number of filters:
    - in encoder blocks: 16, 32, 64.
    - in decoder blocks: 32, 16, 8.
    Default number of classes: 2.
    Default output activation: "softmax"
    """

    def __init__(self,
                 num_encoder_filters=(16, 32, 64),
                 num_decoder_filters=(32, 16, 8),
                 num_classes=2,
                 output_activation="softmax"):

        super(KiUNet3, self).__init__()

        # Fill model configuration parameters
        self.num_encoder_filters = num_encoder_filters
        self.num_decoder_filters = num_decoder_filters
        self.num_classes = num_classes
        self.output_activation = output_activation

        # Instantiate UNet model layers
        self.unet_encoder_conv_1 = Conv2DBn(filter_num=num_encoder_filters[0],
                                            kernel_size=(3, 3))

        self.unet_max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.unet_encoder_conv_2 = Conv2DBn(filter_num=num_encoder_filters[1],
                                            kernel_size=(3, 3))

        self.unet_max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.unet_encoder_conv_3 = Conv2DBn(filter_num=num_encoder_filters[2],
                                            kernel_size=(3, 3))

        self.unet_max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.unet_decoder_conv_1 = Conv2DBn(filter_num=num_decoder_filters[0],
                                            kernel_size=(3, 3))

        self.unet_up_sampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.unet_decoder_conv_2 = Conv2DBn(filter_num=num_decoder_filters[1],
                                            kernel_size=(3, 3))

        self.unet_up_sampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.unet_decoder_conv_3 = Conv2DBn(filter_num=num_decoder_filters[2],
                                            kernel_size=(3, 3))

        self.unet_up_sampling_3 = tf.keras.layers.UpSampling2D(size=(2, 2))

        # Instantiate KiteNet model layers
        self.kite_net_encoder_conv_1 = Conv2DBn(filter_num=num_encoder_filters[0],
                                                kernel_size=(3, 3))

        self.kite_net_up_sampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                                   interpolation="bilinear")

        self.kite_net_encoder_conv_2 = Conv2DBn(filter_num=num_encoder_filters[1],
                                                kernel_size=(3, 3))

        self.kite_net_up_sampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                                   interpolation="bilinear")

        self.kite_net_encoder_conv_3 = Conv2DBn(filter_num=num_encoder_filters[2],
                                                kernel_size=(3, 3))

        self.kite_net_up_sampling_3 = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                                   interpolation="bilinear")

        self.kite_net_decoder_conv_1 = Conv2DBn(filter_num=num_decoder_filters[0],
                                                kernel_size=(3, 3))

        self.kite_net_max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.kite_net_decoder_conv_2 = Conv2DBn(filter_num=num_decoder_filters[1],
                                                kernel_size=(3, 3))

        self.kite_net_max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=2)

        self.kite_net_decoder_conv_3 = Conv2DBn(filter_num=num_decoder_filters[2],
                                                kernel_size=(3, 3))

        self.kite_net_max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=2)

        # Instantiate layers of first CRBF block of encoder
        self.unet_crfb_conv_1 = Conv2DBn(filter_num=num_encoder_filters[0],
                                         kernel_size=(3, 3))

        self.unet_crfb_up_sampling_1 = tf.keras.layers.UpSampling2D(size=(4, 4),
                                                                    interpolation="bilinear")

        self.kite_net_crfb_conv_1 = Conv2DBn(filter_num=num_encoder_filters[0],
                                             kernel_size=(3, 3))

        self.kite_net_crfb_avg_pool_1 = tf.keras.layers.AveragePooling2D(pool_size=4)

        # Instantiate layers of second CRBF block of encoder
        self.unet_crfb_conv_2 = Conv2DBn(filter_num=num_encoder_filters[1],
                                         kernel_size=(3, 3))

        self.unet_crfb_up_sampling_2 = tf.keras.layers.UpSampling2D(size=(16, 16),
                                                                    interpolation="bilinear")

        self.kite_net_crfb_conv_2 = Conv2DBn(filter_num=num_encoder_filters[1],
                                             kernel_size=(3, 3))

        self.kite_net_crfb_avg_pool_2 = tf.keras.layers.AveragePooling2D(pool_size=16)

        # Instantiate layers of third CRBF block of encoder
        self.unet_crfb_conv_3 = Conv2DBn(filter_num=num_encoder_filters[2],
                                         kernel_size=(3, 3))

        self.unet_crfb_up_sampling_3 = tf.keras.layers.UpSampling2D(size=(64, 64),
                                                                    interpolation="bilinear")

        self.kite_net_crfb_conv_3 = Conv2DBn(filter_num=num_encoder_filters[2],
                                             kernel_size=(3, 3))

        self.kite_net_crfb_avg_pool_3 = tf.keras.layers.AveragePooling2D(pool_size=64)

        # Instantiate layers of first CRBF block of decoder
        self.unet_crfb_conv_4 = Conv2DBn(filter_num=num_decoder_filters[0],
                                         kernel_size=(3, 3))

        self.unet_crfb_up_sampling_4 = tf.keras.layers.UpSampling2D(size=(16, 16),
                                                                    interpolation="bilinear")

        self.kite_net_crfb_conv_4 = Conv2DBn(filter_num=num_decoder_filters[0],
                                             kernel_size=(3, 3))

        self.kite_net_crfb_avg_pool_4 = tf.keras.layers.AveragePooling2D(pool_size=16)

        # Instantiate layers of second CRBF block of decoder
        self.unet_crfb_conv_5 = Conv2DBn(filter_num=num_decoder_filters[1],
                                         kernel_size=(3, 3))

        self.unet_crfb_up_sampling_5 = tf.keras.layers.UpSampling2D(size=(4, 4),
                                                                    interpolation="bilinear")

        self.kite_net_crfb_conv_5 = Conv2DBn(filter_num=num_decoder_filters[1],
                                             kernel_size=(3, 3))

        self.kite_net_crfb_avg_pool_5 = tf.keras.layers.AveragePooling2D(pool_size=4)

        # Instantiate layers of the head
        self.head_conv = tf.keras.layers.Conv2D(filters=num_classes,
                                                kernel_size=(1, 1))

        self.final_activation = tf.keras.layers.Activation(activation=output_activation)

    def get_config(self):
        model_configuration = {"num_encoder_filters": self.num_encoder_filters,
                               "num_decoder_filters": self.num_decoder_filters,
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

        # First level of UNet encoder
        x_unet = self.unet_encoder_conv_1(inputs, training=training)
        x_unet_skip_connection_1 = self.unet_max_pool_1(x_unet)

        # First level of KiteNet encoder
        x_kite_net = self.kite_net_encoder_conv_1(inputs, training=training)
        x_kite_net_skip_connection_1 = self.kite_net_up_sampling_1(x_kite_net)

        # CRFB of fist encoder level
        x_unet_crfb = self.unet_crfb_conv_1(x_unet_skip_connection_1, training=training)
        x_unet_crfb = self.unet_crfb_up_sampling_1(x_unet_crfb)

        x_kite_net_crbf = self.kite_net_crfb_conv_1(x_kite_net_skip_connection_1, training=training)
        x_kite_net_crbf = self.kite_net_crfb_avg_pool_1(x_kite_net_crbf)

        x_kite_net = tf.keras.layers.add([x_unet_crfb, x_kite_net_skip_connection_1])
        x_unet = tf.keras.layers.add([x_kite_net_crbf, x_unet_skip_connection_1])

        # Second level of UNet encoder
        x_unet = self.unet_encoder_conv_2(x_unet, training=training)
        x_unet_skip_connection_2 = self.unet_max_pool_2(x_unet)

        # Second level of KiteNet encoder
        x_kite_net = self.kite_net_encoder_conv_2(x_kite_net, training=training)
        x_kite_net_skip_connection_2 = self.kite_net_up_sampling_2(x_kite_net)

        # CRFB of second encoder level
        x_unet_crfb = self.unet_crfb_conv_2(x_unet_skip_connection_2, training=training)
        x_unet_crfb = self.unet_crfb_up_sampling_2(x_unet_crfb)

        x_kite_net_crbf = self.kite_net_crfb_conv_2(x_kite_net_skip_connection_2, training=training)
        x_kite_net_crbf = self.kite_net_crfb_avg_pool_2(x_kite_net_crbf)

        x_kite_net = tf.keras.layers.add([x_unet_crfb, x_kite_net_skip_connection_2])
        x_unet = tf.keras.layers.add([x_kite_net_crbf, x_unet_skip_connection_2])

        # Third level of UNet encoder
        x_unet = self.unet_encoder_conv_3(x_unet, training=training)
        x_unet = self.unet_max_pool_3(x_unet)

        # Third level of KiteNet encoder
        x_kite_net = self.kite_net_encoder_conv_3(x_kite_net, training=training)
        x_kite_net = self.kite_net_up_sampling_3(x_kite_net)

        # CRFB of third encoder level
        x_unet_crfb = self.unet_crfb_conv_3(x_unet, training=training)
        x_unet_crfb = self.unet_crfb_up_sampling_3(x_unet_crfb)

        x_kite_net_crbf = self.kite_net_crfb_conv_3(x_kite_net, training=training)
        x_kite_net_crbf = self.kite_net_crfb_avg_pool_3(x_kite_net_crbf)

        x_kite_net = tf.keras.layers.add([x_unet_crfb, x_kite_net])
        x_unet = tf.keras.layers.add([x_kite_net_crbf, x_unet])

        # First level of UNet decoder
        x_unet = self.unet_decoder_conv_1(x_unet, training=training)
        x_unet = self.unet_up_sampling_1(x_unet)

        # First level of KiteNet decoder
        x_kite_net = self.kite_net_decoder_conv_1(x_kite_net, training=training)
        x_kite_net = self.kite_net_max_pool_1(x_kite_net)

        # CRFB of first decoder level
        x_unet_crfb = self.unet_crfb_conv_4(x_unet, training=training)
        x_unet_crfb = self.unet_crfb_up_sampling_4(x_unet_crfb)

        x_kite_net_crbf = self.kite_net_crfb_conv_4(x_kite_net, training=training)
        x_kite_net_crbf = self.kite_net_crfb_avg_pool_4(x_kite_net_crbf)

        x_kite_net = tf.keras.layers.add([x_unet_crfb, x_kite_net])
        x_unet = tf.keras.layers.add([x_kite_net_crbf, x_unet])

        # Skip connections after first decoder level
        x_unet = tf.keras.layers.add([x_unet_skip_connection_2, x_unet])
        x_kite_net = tf.keras.layers.add([x_kite_net_skip_connection_2, x_kite_net])

        # Second level of UNet decoder
        x_unet = self.unet_decoder_conv_2(x_unet, training=training)
        x_unet = self.unet_up_sampling_2(x_unet)

        # Second level of KiteNet decoder
        x_kite_net = self.kite_net_decoder_conv_2(x_kite_net, training=training)
        x_kite_net = self.kite_net_max_pool_2(x_kite_net)

        # CRFB of second decoder level
        x_unet_crfb = self.unet_crfb_conv_5(x_unet, training=training)
        x_unet_crfb = self.unet_crfb_up_sampling_5(x_unet_crfb)

        x_kite_net_crbf = self.kite_net_crfb_conv_5(x_kite_net, training=training)
        x_kite_net_crbf = self.kite_net_crfb_avg_pool_5(x_kite_net_crbf)

        x_kite_net = tf.keras.layers.add([x_unet_crfb, x_kite_net])
        x_unet = tf.keras.layers.add([x_kite_net_crbf, x_unet])

        # Skip connections after first decoder level
        x_unet = tf.keras.layers.add([x_unet_skip_connection_1, x_unet])
        x_kite_net = tf.keras.layers.add([x_kite_net_skip_connection_1, x_kite_net])

        # Third level of UNet decoder
        x_unet = self.unet_decoder_conv_3(x_unet, training=training)
        x_unet = self.unet_up_sampling_3(x_unet)

        # Third level of KiteNet decoder
        x_kite_net = self.kite_net_decoder_conv_3(x_kite_net, training=training)
        x_kite_net = self.kite_net_max_pool_3(x_kite_net)

        # Head
        x = tf.keras.layers.add([x_unet, x_kite_net])
        x = self.head_conv(x)
        output = self.final_activation(x)

        return output
