from lighter.criterion import BaseCriterion
import torch.nn.functional as F


class Criterion(BaseCriterion):
    def forward(self, inputs, target):
        return F.binary_cross_entropy(inputs, target)
