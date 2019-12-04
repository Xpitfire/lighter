import torch

from lighter.decorator import context, device


class BaseModule(torch.nn.Module):
    """
    Base model class descending from a PyTorch Module base class with injected 'context' instances.
    """
    @device
    @context
    def __init__(self):
        super(BaseModule, self).__init__()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
