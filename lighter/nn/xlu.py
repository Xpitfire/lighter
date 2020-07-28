import torch
import math
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair, _single


class Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = math.sqrt(3) / math.sqrt(self.in_features)
        init.uniform_(self.weight, a=-bound, b=bound)
        if self.bias is not None:
            init.constant_(self.bias, 0)

    def forward(self, input):
        weight = self.weight * torch.exp(-self.weight**2)
        bias = None
        if self.bias is not None:
            bias = self.bias * torch.exp(-self.bias**2)
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.weight * torch.exp(-self.weight**2)
        bias = self.bias
        if bias is not None:
            bias = bias * torch.exp(-bias**2)
        h = self._conv_forward(input, weight, bias)
        return h


class Conv1D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.weight * torch.exp(-self.weight**2)
        bias = self.bias
        if bias is not None:
            bias = bias * torch.exp(-bias**2)
        h = self._conv_forward(input, weight, bias)
        return h
