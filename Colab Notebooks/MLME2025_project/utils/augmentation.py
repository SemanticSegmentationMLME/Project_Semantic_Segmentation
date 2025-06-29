from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import random
from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


# class to perform the random horizontal flip both on images and labels with probability p
class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)

        return img, label


# class to perform the random crop with a specific size both on images and labels with probability p
class RandomCropPair:
    def __init__(self, size, p=0.5):
        self.size = size  
        self.p = p        

    def __call__(self, img, label):
        if random.random() < self.p:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.size)
            img = F.crop(img, i, j, h, w)
            label = F.crop(label, i, j, h, w)

        return img, label


# class to add a Gaussian noise to the image with probability p
class AdditiveGaussianNoise(object):
    def __init__(self, mean=0., std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0., 1.)

        return tensor