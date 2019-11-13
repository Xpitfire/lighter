import torch
from torchvision import models

from lighter.decorator import config
from lighter.model import BaseModule


class Model(BaseModule):
    @config(path='minerl/models/defaults.config.json', property='model')
    def __init__(self):
        super(Model, self).__init__()
        # TODO: setup network architecture for forward pass

    def forward(self, x):
        # TODO: setup forward pass
        raise NotImplementedError()
