import tensorflow as tf
from tf_models.BaseModel import BaseModel
from tf_models.backbones_collection import backbones_collection


class UNet(BaseModel):

    """
    This class defines UNet model that can be used with pretrained backbones
    """

    def __init__(self):

        """
        This method initializes parameters
        :return: None
        """

        super(UNet, self).__init__()

        self.input_shape = (512, 512, 3)

        self.backbone = None

        self.encoder_conv_filters = [32, 64, 128, 256, 512]
        self.encoder_kernels_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        self.encoder_pools_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]

        self.bottleneck_conv_filters = 1024
        self.bottleneck_conv_kernel_size = (3, 3)

        self.decoder_pooling_ratios = [16, 8, 4, 2, 1]
        self.decoder_conv_filters = [512, 256, 128, 64, 32]
        self.decoder_ups_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
        self.decoder_kernels_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        self.decoder_up_conv_kernels_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]

        self.activation = "relu"

        self.head_up_size = (2, 2)
        self.head_conv_filters = 32
        self.head_conv_kernel_size = (3, 3)

        self.classes_num = 1
        self.output_kernel_size = (1, 1)
        self.output_activation = "sigmoid"

        self.skip_connections = {
            1: "act2_encoder_block0",
            2: "act2_encoder_block1",
            4: "act2_encoder_block2",
            8: "act2_encoder_block3",
            16: "act2_encoder_block4",
        }

    def parse_args(self, **kwargs):

        """
        This method sets values of class parameters that exist in kwargs
        :param kwargs: dictionary that contains values of parameters to be set
        :return: None
        """

        super(UNet, self).parse_args(**kwargs)

        params = kwargs["params"]

        if "input_shape" in params.keys():
            self.input_shape = params["input_shape"]

        if "backbone" in params.keys():
            self.backbone = params["backbone"]

        if "encoder_conv_filters" in params.keys():
            self.encoder_conv_filters = params["encoder_conv_filters"]

        if "encoder_kernels_size" in params.keys():
            self.encoder_kernels_size = params["encoder_kernels_size"]

        if "bottleneck_conv_filters" in params.keys():
            self.bottleneck_conv_filters = params["bottleneck_conv_filters"]

        if "bottleneck_conv_kernel_size" in params.keys():
            self.bottleneck_conv_kernel_size = params["bottleneck_conv_kernel_size"]

        if "decoder_pooling_ratios" in params.keys():
            self.decoder_pooling_ratios = params["decoder_pooling_ratios"]

        if "decoder_conv_filters" in params.keys():
            self.decoder_conv_filters = params["decoder_conv_filters"]

        if "decoder_ups_size" in params.keys():
            self.decoder_ups_size = params["decoder_ups_size"]

        if "decoder_kernels_size" in params.keys():
            self.decoder_kernels_size = params["decoder_kernels_size"]

        if "decoder_up_conv_kernels_size" in params.keys():
            self.decoder_up_conv_kernels_size = params["decoder_up_conv_kernels_size"]

        if "activation" in params.keys():
            self.activation = params["activation"]

        if "head_up_size" in params.keys():
            self.head_up_size = params["head_up_size"]

        if "head_conv_filters" in params.keys():
            self.head_conv_filters = params["head_conv_filters"]

        if "head_conv_kernel_size" in params.keys():
            self.head_conv_kernel_size = params["head_conv_kernel_size"]

        if "classes_num" in params.keys():
            self.classes_num = params["classes_num"]

        if "output_kernel_size" in params.keys():
            self.output_kernel_size = params["output_kernel_size"]

        if "output_activation" in params.keys():
            self.output_activation = params["output_activation"]

    def _get_encoder(self):

        """
        This method builds standard UNet encoder with depth defined by parameters
        :return: encoder model
        """

        inputs = tf.keras.layers.Input(self.input_shape)
        x = tf.keras.layers.Lambda(lambda y: y)(inputs)  # helps to proper connect layers inside loop

        # Build encoder
        for level_idx in range(len(self.decoder_pooling_ratios)):

            x = tf.keras.layers.Conv2D(filters=self.encoder_conv_filters[level_idx],
                                       kernel_size=self.encoder_kernels_size[level_idx],
                                       padding='same',
                                       kernel_initializer=self.kernel_initializer,
                                       kernel_regularizer=self._get_regularizer(),
                                       bias_regularizer=self._get_regularizer(),
                                       name="conv1_encoder_block" + str(level_idx))(x)

            x = tf.keras.layers.BatchNormalization(axis=3,
                                                   name="bn1_encoder_block" + str(level_idx))(x)

            x = tf.keras.layers.Activation(activation=self.activation,
                                           name="act1_encoder_block" + str(level_idx))(x)

            x = tf.keras.layers.Conv2D(filters=self.encoder_conv_filters[level_idx],
                                       kernel_size=self.encoder_kernels_size[level_idx],
                                       padding='same',
                                       kernel_initializer=self.kernel_initializer,
                                       kernel_regularizer=self._get_regularizer(),
                                       bias_regularizer=self._get_regularizer(),
                                       name="conv2_encoder_block" + str(level_idx))(x)

            x = tf.keras.layers.BatchNormalization(axis=3,
                                                   name="bn2_encoder_block" + str(level_idx))(x)

            x = tf.keras.layers.Activation(activation=self.activation,
                                           name="act2_encoder_block" + str(level_idx))(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=self.encoder_pools_size[level_idx],
                                             name="pool_encoder_block" + str(level_idx))(x)

        # Build bottleneck layer

        x = tf.keras.layers.Conv2D(filters=self.bottleneck_conv_filters,
                                   kernel_size=self.bottleneck_conv_kernel_size,
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self._get_regularizer(),
                                   bias_regularizer=self._get_regularizer(),
                                   name="conv1_bottleneck")(x)

        x = tf.keras.layers.BatchNormalization(axis=3,
                                               name="bn1_bottleneck")(x)

        x = tf.keras.layers.Activation(activation=self.activation,
                                       name="act1_bottleneck")(x)

        x = tf.keras.layers.Conv2D(filters=self.bottleneck_conv_filters,
                                   kernel_size=self.bottleneck_conv_kernel_size,
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self._get_regularizer(),
                                   bias_regularizer=self._get_regularizer(),
                                   name="conv2_bottleneck")(x)

        x = tf.keras.layers.BatchNormalization(axis=3,
                                               name="bn2_bottleneck")(x)

        outputs = tf.keras.layers.Activation(activation=self.activation,
                                             name="act2_bottleneck")(x)

        encoder = tf.keras.Model(inputs=inputs, outputs=outputs)

        return encoder

    def _get_skip_connection(self, encoder, pooling_ratio):

        """
        This method returns output of skip connection layer for each pooling ratio
        :param encoder: model that contains pooling layers
        :param pooling_ratio: pooling ratio for which skip connections will be returned
        :return: output of skip connection layer
        """

        layer_name = self.skip_connections[pooling_ratio]
        layer = encoder.get_layer(name=layer_name)

        return layer.output

    def build(self):

        """
        This method builds architecture of UNet model with / without backbone
        :return: None
        """

        # Get encoder and skip connections
        if self.backbone is not None:

            backbone = backbones_collection[self.backbone]()

            self.params["pooling"] = None
            backbone.parse_args(params=self.params)
            encoder = backbone.get_backbone()

            skip_connections = [backbone.get_skip_connection(encoder, pooling_ratio) for pooling_ratio in self.decoder_pooling_ratios]

        else:

            encoder = self._get_encoder()
            skip_connections = [self._get_skip_connection(encoder, pooling_ratio) for pooling_ratio in self.decoder_pooling_ratios]

        # Build decoder
        x = encoder.output
        for skip_connection_idx, skip_connection in enumerate(skip_connections):

            # The skip connection of ResNet50V2 model is an convolution layer that is not proceeded by batch normalization and activation
            if skip_connection_idx > 0 and self.backbone == "resnet_50_v2":
                x = tf.keras.layers.BatchNormalization(axis=3,
                                                       name="skip_bn_decoder_block" + str(skip_connection_idx))(x)

                x = tf.keras.layers.Activation(activation=self.activation,
                                               name="skip_act_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.UpSampling2D(size=self.decoder_ups_size[skip_connection_idx],
                                             name="up_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.Conv2D(filters=self.decoder_conv_filters[skip_connection_idx],
                                       kernel_size=self.decoder_up_conv_kernels_size[skip_connection_idx],
                                       padding='same',
                                       kernel_initializer=self.kernel_initializer,
                                       kernel_regularizer=self._get_regularizer(),
                                       bias_regularizer=self._get_regularizer(),
                                       name="up_conv_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.BatchNormalization(axis=3,
                                                   name="up_bn_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.Activation(activation=self.activation,
                                           name="up_act_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.Concatenate(axis=3,
                                            name="concat_decoder_block" + str(skip_connection_idx))([x, skip_connection])

            x = tf.keras.layers.Conv2D(filters=self.decoder_conv_filters[skip_connection_idx],
                                       kernel_size=self.decoder_kernels_size[skip_connection_idx],
                                       padding='same',
                                       kernel_initializer=self.kernel_initializer,
                                       kernel_regularizer=self._get_regularizer(),
                                       bias_regularizer=self._get_regularizer(),
                                       name="conv1_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.BatchNormalization(axis=3,
                                                   name="bn1_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.Activation(activation=self.activation,
                                           name="act1_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.Conv2D(filters=self.decoder_conv_filters[skip_connection_idx],
                                       kernel_size=self.decoder_kernels_size[skip_connection_idx],
                                       padding='same',
                                       kernel_initializer=self.kernel_initializer,
                                       kernel_regularizer=self._get_regularizer(),
                                       bias_regularizer=self._get_regularizer(),
                                       name="conv2_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.BatchNormalization(axis=3,
                                                   name="bn2_decoder_block" + str(skip_connection_idx))(x)

            x = tf.keras.layers.Activation(activation=self.activation,
                                           name="act2_decoder_block" + str(skip_connection_idx))(x)

        # Upsample to original size if it wasn't done in decoder
        if self.decoder_pooling_ratios[-1] != 1:

            x = tf.keras.layers.UpSampling2D(size=self.head_up_size,
                                             name="up_head_block")(x)
        # Build head
        x = tf.keras.layers.Conv2D(filters=self.head_conv_filters,
                                   kernel_size=self.head_conv_kernel_size,
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self._get_regularizer(),
                                   bias_regularizer=self._get_regularizer(),
                                   name="conv1_head_block")(x)

        x = tf.keras.layers.BatchNormalization(axis=3,
                                               name="bn1_head_block")(x)

        x = tf.keras.layers.Activation(activation=self.activation,
                                       name="act1_head_block")(x)

        x = tf.keras.layers.Conv2D(filters=self.head_conv_filters,
                                   kernel_size=self.head_conv_kernel_size,
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self._get_regularizer(),
                                   bias_regularizer=self._get_regularizer(),
                                   name="conv2_head_block")(x)

        x = tf.keras.layers.BatchNormalization(axis=3,
                                               name="bn2_head_block")(x)

        x = tf.keras.layers.Activation(activation=self.activation,
                                       name="act2_head_block")(x)

        x = tf.keras.layers.Conv2D(filters=self.classes_num,
                                   kernel_size=self.output_kernel_size,
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self._get_regularizer(),
                                   bias_regularizer=self._get_regularizer(),
                                   name="output_conv")(x)

        outputs = tf.keras.layers.Activation(activation=self.output_activation,
                                             name="output_act")(x)

        self.model = tf.keras.Model(inputs=encoder.input, outputs=outputs)
