import torch

from lighter.decorator import context


class BaseModule(torch.nn.Module):
    """
    Base model class descending from a PyTorch Module base class with injected 'context' instances.
    """
    @context
    def __init__(self):
        super(BaseModule, self).__init__()
