import torch
from torchvision import models

from lighter.decorator import config
from lighter.model import BaseModule


class AlexNetFeatureExtractionModel(BaseModule):
    @config(path='examples/coco_looking/models/defaults.config.json', group='model')
    def __init__(self):
        super(AlexNetFeatureExtractionModel, self).__init__()
        self.alexnet = models.alexnet(pretrained=self.config.model.pretrained).to(self.device)
        for param in self.alexnet.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.final = torch.nn.Linear(self.alexnet.classifier[-1].out_features, self.config.model.output).to(self.device)

    def forward(self, x):
        h = self.alexnet(x)
        h = self.final(h)
        return torch.sigmoid(h)
