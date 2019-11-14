import torch
from torchvision import models

from lighter.decorator import config, search
from lighter.model import BaseModule
from lighter.parameter import GridParameter


class AlexNetFeatureExtractionModel(BaseModule):
    @config(path='models/defaults.config.json', property='model')
    @search(params=[('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=1000, step=100))])
    def __init__(self):
        super(AlexNetFeatureExtractionModel, self).__init__()
        self.alexnet = models.alexnet(pretrained=self.config.model.pretrained).to(self.device)
        for param in self.alexnet.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.hidden = torch.nn.Linear(self.alexnet.classifier[-1].out_features, self.config.model.hidden_units)
        self.final = torch.nn.Linear(self.config.model.hidden_units, self.config.model.output).to(self.device)

    def forward(self, x):
        h = self.alexnet(x)
        h = self.final(h)
        return torch.sigmoid(h)