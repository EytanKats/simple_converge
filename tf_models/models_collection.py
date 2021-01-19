from tf_models.BaseModel import BaseModel

from tf_models.custom_models.ResNet import ResNet as ResNet_Custom
from tf_models.custom_models.UNet import UNet as UNet_Custom
from tf_models.custom_models.KiUNet import KiUNet as KiUNet_Custom

models_collection = {

    "base_model": BaseModel,

    "resnet_custom": ResNet_Custom,
    "unet_custom": UNet_Custom,
    "kiu_net_custom": KiUNet_Custom

}
