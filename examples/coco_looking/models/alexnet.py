import torch
import torch.nn as nn
from torchvision import models
from lighter.decorator import config
from lighter.model import BaseModule


class AlexNetFeatureExtractionModel(BaseModule):
    @config(path='models/alexnet.config.json', property='model')
    def __init__(self):
        super(AlexNetFeatureExtractionModel, self).__init__()
        self.alexnet = models.alexnet(pretrained=self.config.model.pretrained)
        for param in self.alexnet.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        previous_layer = self.alexnet.classifier._modules['4']
        self.alexnet.classifier._modules['6'] = nn.Linear(previous_layer.out_features, self.config.model.output)
        self.alexnet = self.alexnet.to(self.device)

    def forward(self, x):
        h = self.alexnet(x)
        return torch.sigmoid(h)
