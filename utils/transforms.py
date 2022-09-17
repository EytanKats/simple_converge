import random
from PIL import ImageFilter

import torch


class GaussianBlur(object):

    def __init__(self, sigma=(1, 1)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianNoise(object):

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, x):
        noise = torch.randn(x.size()) * self.std + self.mean
        x = x + noise
        return x
