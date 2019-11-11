import torch
from torchvision import models

from lighter.decorator import config
from lighter.model import BaseModule


class Network(BaseModule):
    @config(path='minerl/models/defaults.config.json', group='model')
    def __init__(self):
        super(Network, self).__init__()
        # TODO: setup network architecture for forward pass

    def forward(self, x):
        # TODO: setup forward pass
        raise NotImplementedError()
