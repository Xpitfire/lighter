import torch
import numpy as np


class XdA(torch.nn.Module):
    """
    Context-dependent activations implementation introduced by Dinu et al. in the thesis
    Overcoming Catastrophic Forgetting with Context-Dependent Activations and Synaptic Stabilization
    https://www.dinu.at/wp-content/uploads/2019/11/Overcoming-Catastrophic-Forgetting-with-Context-Dependent-Activations-and-Synaptic-Stabilization.pdf
    track the layer activation statistics and use the summary statistics for the gating non-linearity.
    All activation functions that are based-on activation space statistics are denoted as context-dependent activations.
    This XdA class implementation currently supports two summary statistic variants, 'mean' and 'median' and the
    gating function is defined as followed:
    f(x) = | x >= phi  --> x (identity)
           | otherwise --> beta * x
    and
    phi <- (1 - alpha) * phi + alpha * x_bar
    whereas phi is the cumulative normalized summary statistic, x_bar the current normalized summary statistics of layer
    activations - which is here updated as an exponential moving mode - x are the input activations and
    beta and alpha are hyperparameters for tuning the updates and gating function.
    This gating function uses LayerNorm to re-normalize the input.
    """
    def __init__(self, alpha: float = 0.01,
                 beta: float = 0.0,
                 mode: str = 'mean',
                 norm_type: str = 'layernorm',
                 mask_type: str = 'mask',
                 mse_lr: float = 1e-2):
        """
        Context-dependent activations implementation using normalized activation space summary statistics
        for the non-linearity definition.
        The activation function initializes itself with the first passed data batch.
        ATTENTION: An XdA layer is NOT stateless, such as ReLU or other activation function. You need to instantiate one
        XdA-layer per output layer fot the gating to properly work.
        :param alpha: The summary statistic is updated via an exponential moving mode. The alpha trades of between how
        much the new current vs the new activation summary statistic contributes.
        :param beta: The beta parameter regulates the leakyness of the gating non-linearity.
        :param mode: The summary statistic
        :param norm_type: Normalize the x values according to the given normalization type before activation
        :param mse_lr: Learning rate for the differential parameters in the 'mse' mode.
        :param mask_type: Select the gating mode.
        """
        super(XdA, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.phi = None
        self.norm = None
        self.norm_type = norm_type
        self.mask_type = mask_type
        self.mse_lr = mse_lr
        assert 0 <= self.alpha <= 1
        assert 0 <= self.beta <= 1
        assert 0 <= self.mse_lr
        assert self.mode in ['zero', 'mean', 'median', 'mse']
        assert self.norm_type in [None, 'vecnorm', 'batchnorm', 'layernorm', 'multinorm', 'mean', 'std']
        assert self.mask_type in ['mask', 'max']

    @staticmethod
    def _multi_norm(x, eps=1e-7):
        """
        Computes the batch and layer norm for the committed value x.
        """
        b_mean = torch.mean(x, dim=0, keepdim=True)
        b_std = torch.sqrt(torch.var(x, dim=0, keepdim=True) + eps)
        x = (x - b_mean) / b_std
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

    @staticmethod
    def _layer_norm(x, eps=1e-7):
        """
        Computes the layer norm for the committed value x.
        """
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

    @staticmethod
    def _batch_norm(x, eps=1e-7):
        """
        Computes the batch norm for the committed value x.
        """
        b_mean = torch.mean(x, dim=0, keepdim=True)
        b_std = torch.sqrt(torch.var(x, dim=0, keepdim=True) + eps)
        x = (x - b_mean) / b_std
        return x

    @staticmethod
    def _vec_norm(x, p=2, eps=1e-7):
        """
        Computes the normalized vector for the committed value x.
        """
        x_shape = x.shape
        if len(x_shape) == 2:
            x_norm = torch.pow(torch.sum(torch.pow(x, p), dim=-1, keepdim=True) + eps, 1/p)
            x = x / x_norm
        elif len(x_shape) > 2:
            x = x.view(x_shape[0], -1)
            x_norm = torch.pow(torch.sum(torch.pow(x, p), dim=-1, keepdim=True) + eps, 1/p)
            x = x / x_norm
            x = x.view(x_shape)
        else:
            raise NotImplementedError('Unsupported normalization shape: {}'.format(x_shape))
        return x

    @staticmethod
    def _mean(x):
        """
        Computes the mean normalization for a given x.
        """
        x_shape = x.shape
        if len(x_shape) == 2:
            x = x - torch.mean(x, dim=1, keepdim=True)
        elif len(x_shape) > 2:
            x = x.view(x_shape[0], -1)
            x = x - torch.mean(x, dim=1, keepdim=True)
            x = x.view(x_shape)
        else:
            raise NotImplementedError('Unsupported normalization shape: {}'.format(x_shape))
        return x

    @staticmethod
    def _std(x, eps=1e-7):
        """
        Computes the mean normalization for a given x.
        """
        x_shape = x.shape
        if len(x_shape) == 2:
            x_std = torch.sqrt(torch.var(x, dim=1, keepdim=True) + eps)
            x = x / x_std
        elif len(x_shape) > 2:
            x = x.view(x_shape[0], -1)
            x_std = torch.sqrt(torch.var(x, dim=1, keepdim=True) + eps)
            x = x / x_std
            x = x.view(x_shape)
        else:
            raise NotImplementedError('Unsupported normalization shape: {}'.format(x_shape))
        return x

    def _check_init(self, x):
        """
        Checks if the activation function is already initialized and if not, it creates the respective
        methods.
        :param x: the activation space inputs
        :return:
        """
        if self.phi is None:
            # init normalization
            if self.norm_type == 'vecnorm':
                self.norm = XdA._vec_norm
            elif self.norm_type == 'batchnorm':
                self.norm = XdA._layer_norm
            elif self.norm_type == 'layernorm':
                self.norm = XdA._layer_norm
            elif self.norm_type == 'multinorm':
                self.norm = XdA._multi_norm
            elif self.norm_type == 'mean':
                self.norm = XdA._mean
            elif self.norm_type == 'std':
                self.norm = XdA._std
            # init phi parameters
            if self.mode == 'zero':
                self.phi = torch.nn.Parameter(torch.zeros(x.shape[1:]), requires_grad=False).to(x.device)
            elif self.mode == 'mean':
                self.phi = torch.nn.Parameter(torch.zeros(x.shape[1:]), requires_grad=False).to(x.device)
                x_ = x.detach().mean(0).data
                self.phi.data = self.phi + x_
            elif self.mode == 'median':
                self.phi = torch.nn.Parameter(torch.zeros(x.shape[1:]), requires_grad=False).to(x.device)
                x_ = torch.median(x.detach(), dim=0)[0]
                self.phi.data = self.phi + x_
            elif self.mode == 'mse':
                self.phi = torch.from_numpy(np.random.randn(*x.shape[1:]).astype(np.float32)).to(x.device)
                self.phi.requires_grad = True
            self.init = False

    def _update_phi(self, x):
        """
        Updates the gating summary statistics according to the defined mode ('mean', 'median').
        :param x: the activation space inputs
        :return:
        """
        if self.training:
            if self.mode == 'mean':
                x_ = x.detach().mean(0)
                self.phi.data = ((1 - self.alpha) * self.phi + self.alpha * x_).data
            elif self.mode == 'median':
                x_ = torch.median(x.detach(), dim=0)[0]
                self.phi.data = ((1 - self.alpha) * self.phi + self.alpha * x_).data
            elif self.mode == 'mse':
                self.phi.data -= self.mse_lr * (self.phi - x.detach()).mean()

    def _mask(self, x):
        """
        Creates a binary mask according to the cumulative summary statistics.
        :param x: teh activation space inputs
        :return: binary mask with same dimensions as the inputs
        """
        return (x >= self.phi.expand_as(x)).float()

    def _gate(self, x):
        if self.mask_type == 'mask':
            # creates a binary mask to disable units
            mask = self._mask(x)
            # disables units
            h = x * mask
            # checks if beta is enabled and applies a leaky variant
            if self.beta != 0:
                h += (self.beta * x * (1.0 - mask))
        elif self.mask_type == 'max':
            if self.beta == 0:
                h = torch.max(x, self.phi.expand_as(x))
            else:
                h = torch.max(x, self.beta * self.phi.expand_as(x))
        else:
            raise NotImplementedError('Invalid option!')
        return h

    def forward(self, x):
        # check if layer is initialized
        self._check_init(x)
        # normalize the layer according to the layer norm
        if self.norm is not None:
            x = self.norm(x)
        # updates the gating summary statistic
        self._update_phi(x)
        h = self._gate(x)
        return h
