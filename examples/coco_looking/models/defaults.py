import torch
from torchvision import models

from lighter.decorator import config
from lighter.model import BaseModule


class AlexNetFeatureExtractionNetwork(BaseModule):
    @config(path='examples/coco_looking/models/defaults.config.json', group='model')
    def __init__(self):
        super(AlexNetFeatureExtractionNetwork, self).__init__()
        self.alexnet = models.alexnet(pretrained=self.config.model.pretrained).to(self.config.model.device)
        for param in self.alexnet.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.final = torch.nn.Linear(self.alexnet.classifier[-1].out_features, self.config.model.output).to(self.config.model.device)

    def forward(self, x):
        h = self.alexnet(x)
        h = self.final(h)
        return torch.sigmoid(h)
