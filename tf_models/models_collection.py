from tf_models.UNet import UNet
from tf_models.ResNet50 import ResNet50
from tf_models.ResNet101 import ResNet101
from tf_models.ResNet152 import ResNet152
from tf_models.ResNet50V2 import ResNet50V2
from tf_models.DenseNet121 import DenseNet121
from tf_models.EfficientNetB5 import EfficientNetB5
from tf_models.EfficientNetB6 import EfficientNetB6
from tf_models.EfficientNetB7 import EfficientNetB7

from tf_models.toy_models.ClassificationNet import ClassificationNet

models_collection = {

    "resnet_50": ResNet50,
    "resnet_101": ResNet101,
    "resnet_152": ResNet152,
    "resnet_50_v2": ResNet50V2,
    "densenet_121": DenseNet121,
    "efficientnet_b5": EfficientNetB5,
    "efficientnet_b6": EfficientNetB6,
    "efficientnet_b7": EfficientNetB7,

    "unet": UNet,

    "toy_classification_net": ClassificationNet

}
