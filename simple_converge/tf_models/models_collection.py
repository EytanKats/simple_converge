import os
os.environ["SM_FRAMEWORK"] = "tf.keras"  # set 'segmentation_models' package to work with 'tf.keras' instead of 'keras'

from simple_converge.tf_models.BaseModel import BaseModel

from simple_converge.tf_models.custom_models.UNet import UNet as UNet_Custom
from simple_converge.tf_models.custom_models.KiUNet import KiUNet as KiUNet_Custom
from simple_converge.tf_models.custom_models.ResNet import ResNet as ResNet_Custom

from simple_converge.tf_models.third_party_models.UNet import UNet as UNet_3rdParty
from simple_converge.tf_models.third_party_models.AttentionUNet import AttentionUNet as AttentionUnet_3rdParty
from simple_converge.tf_models.third_party_models.Classifier import Classifier as Classifier_3rdParty

models_collection = {

    "base_model": BaseModel,

    "resnet_custom": ResNet_Custom,
    "unet_custom": UNet_Custom,
    "kiu_net_custom": KiUNet_Custom,

    "unet_3rd_party": UNet_3rdParty,
    "attention_unet_3rd_party": AttentionUnet_3rdParty,
    "classifier_3rd_party": Classifier_3rdParty
}
