# https://torchdrug.ai/docs/_modules/torchdrug/layers/common.html#GaussianSmearing
import torch
import torch.nn as nn

class GaussianSmearing(nn.Module):
    r"""
    Gaussian smearing from
    `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions`_.

    There are two modes for Gaussian smearing.

    Non-centered mode:

    .. math::

        \mu = [0, 1, ..., n], \sigma = [1, 1, ..., 1]

    Centered mode:

    .. math::

        \mu = [0, 0, ..., 0], \sigma = [0, 1, ..., n]

    .. _SchNet\: A continuous-filter convolutional neural network for modeling quantum interactions:
        https://arxiv.org/pdf/1706.08566.pdf

    Parameters:
        start (int, optional): minimal input value
        stop (int, optional): maximal input value
        num_kernel (int, optional): number of RBF kernels
        centered (bool, optional): centered mode or not
        learnable (bool, optional): learnable gaussian parameters or not
    """

    def __init__(self, start=0, stop=5, num_kernel=100, centered=False, learnable=False):
        super(GaussianSmearing, self).__init__()
        if centered:
            mu = torch.zeros(num_kernel)
            sigma = torch.linspace(start, stop, num_kernel)
        else:
            mu = torch.linspace(start, stop, num_kernel)
            sigma = torch.ones(num_kernel) * (mu[1] - mu[0])

        if learnable:
            self.mu = nn.Parameter(mu)
            self.sigma = nn.Parameter(sigma)
        else:
            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)

    def forward(self, x, y):
        """
        Compute smeared gaussian features between data.

        Parameters:
            x (Tensor): data of shape :math:`(..., d)`
            y (Tensor): data of shape :math:`(..., d)`
        Returns:
            Tensor: features of shape :math:`(..., num\_kernel)`
        """
        distance = (x - y).norm(2, dim=-1, keepdim=True)
        z = (distance - self.mu) / self.sigma
        prob = torch.exp(-0.5 * z * z)
        return prob