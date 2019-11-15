import torch
from torchvision import models

from lighter.decorator import config, search
from lighter.model import BaseModule
from lighter.parameter import GridParameter


class ResNetFeatureExtractionModel(BaseModule):
    @config(path='models/resnet.config.json', property='model')
    @search(params=[('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=1000, step=100))])
    def __init__(self):
        super(ResNetFeatureExtractionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=self.config.model.pretrained).to(self.device)
        for param in self.resnet.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.hidden = torch.nn.Linear(self.resnet.fc.out_features, self.config.model.hidden_units).to(self.device)
        self.final = torch.nn.Linear(self.config.model.hidden_units, self.config.model.output).to(self.device)

    def forward(self, x):
        h = self.resnet(x)
        h = self.hidden(h)
        h = self.final(h)
        return torch.sigmoid(h)