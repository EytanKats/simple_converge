import tensorflow as tf
from simple_converge.tf_models.core.ResidualBlock import ResidualBlock


class ResNet18(tf.keras.Model):

    """
    This class implements ResNet18 architecture.
    """

    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        # Fill model configuration parameters
        self.num_classes = num_classes

        # Instantiate model layers
        self.conv = tf.keras.layers.Conv2D(filters=64,
                                           kernel_size=(7, 7),
                                           strides=(2, 2),
                                           padding="same")

        self.bn = tf.keras.layers.BatchNormalization()

        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                  strides=2,
                                                  padding="same")

        self.res_block_1_layer_1 = ResidualBlock(filter_num=64,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1))

        self.res_block_2_layer_1 = ResidualBlock(filter_num=64,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1))

        self.res_block_1_layer_2 = ResidualBlock(filter_num=128,
                                                 kernel_size=(3, 3),
                                                 strides=(2, 2))

        self.res_block_2_layer_2 = ResidualBlock(filter_num=128,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1))

        self.res_block_1_layer_3 = ResidualBlock(filter_num=256,
                                                 kernel_size=(3, 3),
                                                 strides=(2, 2))

        self.res_block_2_layer_3 = ResidualBlock(filter_num=256,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1))

        self.res_block_1_layer_4 = ResidualBlock(filter_num=512,
                                                 kernel_size=(3, 3),
                                                 strides=(2, 2))

        self.res_block_2_layer_4 = ResidualBlock(filter_num=512,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1))

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

    def get_config(self):
        model_configuration = {"num_classes": self.num_classes}
        return model_configuration

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    def call(self,
             inputs,
             training=False,
             mask=None):

        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        x = self.res_block_1_layer_1(x, training=training)
        x = self.res_block_2_layer_1(x, training=training)
        x = self.res_block_1_layer_2(x, training=training)
        x = self.res_block_2_layer_2(x, training=training)
        x = self.res_block_1_layer_3(x, training=training)
        x = self.res_block_2_layer_3(x, training=training)
        x = self.res_block_1_layer_4(x, training=training)
        x = self.res_block_2_layer_4(x, training=training)
        x = self.global_avg_pool(x)
        output = self.fc(x)

        return output