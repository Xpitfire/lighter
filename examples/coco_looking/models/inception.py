import torch
from torchvision import models

from lighter.decorator import config, search
from lighter.model import BaseModule
from lighter.parameter import GridParameter


class InceptionNetFeatureExtractionModel(BaseModule):
    @config(path='models/inception.config.json', property='model')
    @search(params=[('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=1000, step=100))])
    def __init__(self):
        super(InceptionNetFeatureExtractionModel, self).__init__()
        self.inception = models.inception_v3(pretrained=self.config.model.pretrained,
                                             aux_logits=self.config.model.aux_logits).to(self.device)
        for param in self.inception.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.hidden = torch.nn.Linear(self.inception.fc.out_features,
                                      self.config.model.hidden_units).to(self.device)
        self.final = torch.nn.Linear(self.config.model.hidden_units,
                                     self.config.model.output).to(self.device)

    def forward(self, x):
        h = self.inception(x)
        h = torch.relu(h)
        h = self.hidden(h)
        h = torch.relu(h)
        h = self.final(h)
        return torch.sigmoid(h)
