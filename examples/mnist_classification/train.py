from main.main import train
from tf_models.models_collection import models_collection

from generator.ClassesGenerator import ClassesGenerator
from examples.mnist_classification.Settings import Settings
from dataset.toy_datasets.MnistDataset import MnistDataset

# Create class instances
settings = Settings()
dataset = MnistDataset()
generator = ClassesGenerator()

# Test
train(settings, dataset, generator, models_collection)