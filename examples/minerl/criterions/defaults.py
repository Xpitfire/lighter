from lighter.criterion import BaseCriterion
import torch.nn.functional as F


class SimpleCriterion(BaseCriterion):
    def forward(self, inputs, target):
        # TODO: use criterion to determine loss
        raise NotImplementedError()
