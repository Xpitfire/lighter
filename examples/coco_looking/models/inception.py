import torch
import torch.nn as nn
from torchvision import models

from lighter.decorator import config
from lighter.model import BaseModule


class InceptionNetFeatureExtractionModel(BaseModule):
    @config(path='models/inception.config.json', property='model')
    def __init__(self):
        super(InceptionNetFeatureExtractionModel, self).__init__()
        self.inception = models.inception_v3(pretrained=self.config.model.pretrained,
                                             aux_logits=self.config.model.aux_logits).to(self.device)
        for param in self.inception.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.inception.fc = nn.Linear(2048, self.config.model.output)
        self.inception = self.inception.to(self.device)

    def forward(self, x):
        h = self.inception(x)
        return torch.sigmoid(h)
