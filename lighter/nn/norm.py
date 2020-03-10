import torch
import torch.nn as nn


class DynLayerNorm(nn.Module):
    """
    Computes the layer norm for the committed value x. The dimensions are automatically established.
    """
    def __init__(self):
        super(DynLayerNorm, self).__init__()

    @staticmethod
    def _layer_norm(x, eps=1e-7):
        x_shape = x.shape
        if len(x_shape) == 2:
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sqrt(torch.var(x, dim=-1, keepdim=True) + eps)
        elif len(x_shape) > 2:
            x = x.view(x_shape[0], -1)
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sqrt(torch.var(x, dim=-1, keepdim=True) + eps)
            x = x.view(x_shape)
        else:
            raise NotImplementedError('Unsupported normalization shape: {}'.format(x_shape))
        return x

    def forward(self, x):
        return self._layer_norm(x)


class DynBatchNorm(nn.Module):
    """
    Computes the batch and layer norm for the committed value x. The dimensions are automatically established.
    """
    def __init__(self):
        super(DynBatchNorm, self).__init__()

    @staticmethod
    def _multi_norm(x, eps=1e-7):
        b_mean = torch.mean(x, dim=0, keepdim=True)
        b_std = torch.sqrt(torch.var(x, dim=0, keepdim=True) + eps)
        x = (x - b_mean) / b_std
        return x

    def forward(self, x):
        return self._multi_norm(x)

