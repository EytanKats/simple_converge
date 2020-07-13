from tf_models.UNet import UNet
from tf_models.ResNet50 import ResNet50
from tf_models.ResNet101 import ResNet101
from tf_models.ResNet152 import ResNet152
from tf_models.ResNet50V2 import ResNet50V2
from tf_models.DenseNet121 import DenseNet121
from tf_models.EfficientNetB5 import EfficientNetB5
from tf_models.EfficientNetB6 import EfficientNetB6
from tf_models.EfficientNetB7 import EfficientNetB7

models_collection = {

    "resnet_50": ResNet50,
    "resnet_101": ResNet101,
    "resnet_152": ResNet152,
    "resnet_50_v2": ResNet50V2,
    "densenet_121": DenseNet121,
    "efficientnet_b5": EfficientNetB5,
    "efficientnet_b6": EfficientNetB6,
    "efficientnet_b7": EfficientNetB7,
    "unet": UNet

}

# apps = tf.keras.applications
# models_collection = {
#
#     # VGG tensorflow models
#     "vgg_16": apps.vgg16.VGG16,
#     "vgg_19": apps.vgg19.VGG19,
#
#     # ResNet tensorflow models
#     "resnet_50": apps.resnet.ResNet50,
#     "resnet_101": apps.resnet.ResNet101,
#     "resnet_152": apps.resnet.ResNet152,
#
#     "resnet_v2_50": apps.resnet_v2.ResNet50V2,
#     "resnet_v2_101": apps.resnet_v2.ResNet101V2,
#     "resnet_v2_152": apps.resnet_v2.ResNet152V2,
#
#     # Inception/Xception tensorflow models
#     "inception_v3": apps.inception_v3.InceptionV3,
#     "inception_resnet_v2": apps.inception_resnet_v2.InceptionResNetV2,
#     "xception": apps.xception.Xception,
#
#     # DenseNet tensorflow models
#     "densenet_121": apps.densenet.DenseNet121,
#     "densenet_169": apps.densenet.DenseNet169,
#     "densenet_201": apps.densenet.DenseNet201,
#
#     # MobileNet tensorflow models
#     "mobilenet": apps.mobilenet.MobileNet,
#     "mobilenet_v2": apps.mobilenet_v2.MobileNetV2
#
# }
#
# preprocess_methods = {
#
#     # VGG tensorflow models
#     "vgg_16_imagenet": apps.vgg16.preprocess_input,
#     "vgg_19_imagenet": apps.vgg19.preprocess_input,
#
#     # ResNet tensorflow models
#     "resnet_50_imagenet": apps.resnet.preprocess_input,
#     "resnet_101_imagenet": apps.resnet.preprocess_input,
#     "resnet_152_imagenet": apps.resnet.preprocess_input,
#
#     "resnet_v2_50_imagenet": apps.resnet_v2.preprocess_input,
#     "resnet_v2_101_imagenet": apps.resnet_v2.preprocess_input,
#     "resnet_v2_152_imagenet": apps.resnet_v2.preprocess_input,
#
#     # Inception/Xception tensorflow models
#     "inception_v3_imagenet": apps.inception_v3.preprocess_input,
#     "inception_resnet_v2_imagenet": apps.inception_resnet_v2.preprocess_input,
#     "xception_imagenet": apps.xception.preprocess_input,
#
#     # DenseNet tensorflow models
#     "densenet_121_imagenet": apps.densenet.preprocess_input,
#     "densenet_169_imagenet": apps.densenet.preprocess_input,
#     "densenet_201_imagenet": apps.densenet.preprocess_input,
#
#     # MobileNet tensorflow models
#     "mobilenet_imagenet": apps.mobilenet.preprocess_input,
#     "mobilenet_v2_imagenet": apps.mobilenet_v2.preprocess_input
#
# }
