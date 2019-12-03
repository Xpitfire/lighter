import torch
import torch.nn as nn
from torchvision import models

from lighter.decorator import config
from lighter.model import BaseModule


class ResNetFeatureExtractionModel(BaseModule):
    @config(path='models/resnet.config.json', property='model')
    def __init__(self):
        super(ResNetFeatureExtractionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=self.config.model.pretrained)
        for param in self.resnet.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.resnet.fc = nn.Linear(512, self.config.model.output)
        self.resnet = self.resnet.to(self.device)

    def forward(self, x):
        h = self.resnet(x)
        return torch.sigmoid(h)
