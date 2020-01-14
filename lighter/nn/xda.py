import torch


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
                 norm_type: str = 'layernorm'):
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
        """
        super(XdA, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.phi = None
        self.norm = None
        self.norm_type = norm_type
        assert 0 <= self.alpha <= 1
        assert 0 <= self.beta <= 1
        assert self.mode in ['mean', 'median']
        assert self.norm_type in [None, 'norm', 'layernorm', 'layernorm_elementwise_affine']

    def _mask(self, x):
        """
        Creates a binary mask according to the cumulative summary statistics.
        :param x: teh activation space inputs
        :return: binary mask with same dimensions as the inputs
        """
        return (x >= self.phi.expand_as(x)).float()

    @staticmethod
    def _norm(x):
        """
        Computes the normalized vector for the committed value x.
        :param x: the activation space inputs
        :return: normalized action space vector
        """
        x_shape = x.shape
        x = x.view(x_shape[0], -1)
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x = x / x_norm
        x = x.view(x_shape)
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
            if self.norm_type == 'norm':
                self.norm = XdA._norm
            elif self.norm_type == 'layernorm':
                self.norm = torch.nn.LayerNorm(x.shape[1:], elementwise_affine=False)
            elif self.norm_type == 'layernorm_elementwise_affine':
                self.norm = torch.nn.LayerNorm(x.shape[1:], elementwise_affine=True)
            # init phi parameters
            self.phi = torch.nn.Parameter(torch.zeros(x.shape[1:]), requires_grad=False).to(x.device)
            if self.mode == 'mean':
                x_ = x.detach().mean(0).data
            elif self.mode == 'median':
                x_ = torch.median(x.detach(), dim=0)[0]
            else:
                raise NotImplementedError()
            self.phi.data = self.phi + x_
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
            elif self.mode == 'median':
                x_ = torch.median(x.detach(), dim=0)[0]
            else:
                raise NotImplementedError()
            self.phi.data = ((1 - self.alpha) * self.phi + self.alpha * x_).data

    def forward(self, x):
        # check if layer is initialized
        self._check_init(x)
        # normalize the layer according to the layer norm
        if self.norm is not None:
            x = self.norm(x)
        # updates the gating summary statistic
        self._update_phi(x)
        # creates a binary mask to disable units
        mask = self._mask(x)
        # disables units
        h = x * mask
        # checks if beta is enabled and applies a leaky variant
        if self.beta > 0:
            h += (self.beta * x * (1.0 - mask))
        return h
