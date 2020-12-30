from tf_models.BaseModel import BaseModel

from tf_models.third_party_models.UNet import UNet as UNet_TF
from tf_models.third_party_models.ResNet50 import ResNet50 as ResNet50_TF
from tf_models.third_party_models.ResNet50V2 import ResNet50V2 as ResNet50V2_TF
from tf_models.third_party_models.ResNet101 import ResNet101 as ResNet101_TF
from tf_models.third_party_models.ResNet152 import ResNet152 as ResNet152_TF
from tf_models.third_party_models.DenseNet121 import DenseNet121 as DenseNet121_TF
from tf_models.third_party_models.EfficientNetB5 import EfficientNetB5 as EfficientNetB5_EfficientNet
from tf_models.third_party_models.EfficientNetB6 import EfficientNetB6 as EfficientNetB6_EfficientNet
from tf_models.third_party_models.EfficientNetB7 import EfficientNetB7 as EfficientNetB7_EfficientNet

from tf_models.custom_models.ResNet import ResNet as ResNet_Custom
from tf_models.custom_models.UNet import UNet as UNet_Custom

models_collection = {

    "base_model": BaseModel,

    "unet_tf": UNet_TF,
    "resnet_50_tf": ResNet50_TF,
    "resnet_101_tf": ResNet101_TF,
    "resnet_152_tf": ResNet152_TF,
    "resnet_50_v2_tf": ResNet50V2_TF,
    "densenet_121_tf": DenseNet121_TF,

    "efficientnet_b5_efficientnet": EfficientNetB5_EfficientNet,
    "efficientnet_b6_efficientnet": EfficientNetB6_EfficientNet,
    "efficientnet_b7_efficientnet": EfficientNetB7_EfficientNet,

    "resnet_custom": ResNet_Custom,
    "unet_custom": UNet_Custom

}
