import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from lighter.decorator import config
from facenet_pytorch import InceptionResnetV1
from lighter.model import BaseModule


class FaceNetExtractionModel(BaseModule):
    @config(path='models/facenet.config.json', property='model')
    def __init__(self):
        super(FaceNetExtractionModel, self).__init__()
        self.facenet = InceptionResnetV1(pretrained=self.config.model.pretrained).to(self.device)
        for param in self.facenet.parameters():
            param.requires_grad = not self.config.model.freeze_pretrained
        self.spectral_norm = self.config.model.spectral_norm
        self._change_facenet_output_dim()
        self.dropout = nn.Dropout(p=self.config.model.dropout)

    def _change_facenet_output_dim(self):
        if not self.spectral_norm:
            self.facenet.last_linear = nn.Linear(1792, self.config.model.output, bias=True).to(self.device)
        else:
            self.facenet.last_linear = spectral_norm(
                nn.Linear(1792, self.config.model.output, bias=True).to(self.device))

    def forward(self, x):
        x = self.facenet.conv2d_1a(x)
        x = self.facenet.conv2d_2a(x)
        x = self.facenet.conv2d_2b(x)
        x = self.facenet.maxpool_3a(x)
        x = self.facenet.conv2d_3b(x)
        x = self.facenet.conv2d_4a(x)
        x = self.facenet.conv2d_4b(x)
        x = self.facenet.repeat_1(x)
        x = self.facenet.mixed_6a(x)
        x = self.facenet.repeat_2(x)
        x = self.facenet.mixed_7a(x)
        x = self.facenet.repeat_3(x)
        x = self.facenet.block8(x)
        x = self.facenet.avgpool_1a(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.facenet.last_linear(x.view(x.shape[0], -1))
        return torch.sigmoid(x)
