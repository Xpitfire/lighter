import torch
import torch.nn as nn
from torchvision import models
from lighter.decorator import config
from lighter.model import BaseModule


class CustomConvModel(BaseModule):
    @config(path='models/alexnet.config.json', property='model')
    def __init__(self):
        super(CustomConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=3, padding=0).to(self.device)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.fc1 = nn.Linear(in_features=4096, out_features=200).to(self.device)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(in_features=200, out_features=1).to(self.device)

    def forward(self, x):
        h = self.conv1(x)
        h = torch.relu(h)
        h = self.conv2(h)
        h = torch.relu(h)
        h = self.conv3(h)
        h = torch.relu(h)
        h = self.conv4(h)
        h = torch.relu(h)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.dropout1(h)
        h = self.fc2(h)
        return torch.sigmoid(h)
