import torch
import torch.nn as nn
from . import mask_utils

__all__ = ['CouplingBlock', 'ChannelCouplingBlock', 'MaskedCouplingBlock']


class CouplingBlock(nn.Module):
    """Basic coupling layer for Tensors of shape (n,d).

    Forward computation:
        y_a = x_a
        y_b = y_b * exp(s(x_a)) + t(x_a)
    Inverse computation:
        x_a = y_a
        x_b = (y_b - t(y_a)) * exp(-s(y_a))
    """

    def __init__(self, dim, nnet, swap=False):
        """
        Args:
            s (nn.Module)
            t (nn.Module)
        """
        super(CouplingBlock, self).__init__()
        assert (dim % 2 == 0)
        self.d = dim // 2
        self.nnet = nnet
        self.swap = swap

    def func_s_t(self, x):
        f = self.nnet(x)
        s = f[:, :self.d]
        t = f[:, self.d:]
        return s, t

    def forward(self, x, logpx=None):
        """Forward computation of a simple coupling split on the axis=1.
        """
        x_a = x[:, :self.d] if not self.swap else x[:, self.d:]
        x_b = x[:, self.d:] if not self.swap else x[:, :self.d]
        y_a, y_b, logdetgrad = self._forward_computation(x_a, x_b)
        y = [y_a, y_b] if not self.swap else [y_b, y_a]

        if logpx is None:
            return torch.cat(y, dim=1)
        else:
            return torch.cat(y, dim=1), logpx - logdetgrad.view(x.size(0), -1).sum(1, keepdim=True)

    def inverse(self, y, logpy=None):
        """Inverse computation of a simple coupling split on the axis=1.
        """
        y_a = y[:, :self.d] if not self.swap else y[:, self.d:]
        y_b = y[:, self.d:] if not self.swap else y[:, :self.d]
        x_a, x_b, logdetgrad = self._inverse_computation(y_a, y_b)
        x = [x_a, x_b] if not self.swap else [x_b, x_a]
        if logpy is None:
            return torch.cat(x, dim=1)
        else:
            return torch.cat(x, dim=1), logpy + logdetgrad

    def _forward_computation(self, x_a, x_b):
        y_a = x_a
        s_a, t_a = self.func_s_t(x_a)
        scale = torch.sigmoid(s_a + 2.)
        y_b = x_b * scale + t_a
        logdetgrad = self._logdetgrad(scale)
        return y_a, y_b, logdetgrad

    def _inverse_computation(self, y_a, y_b):
        x_a = y_a
        s_a, t_a = self.func_s_t(y_a)
        scale = torch.sigmoid(s_a + 2.)
        x_b = (y_b - t_a) / scale
        logdetgrad = self._logdetgrad(scale)
        return x_a, x_b, logdetgrad

    def _logdetgrad(self, scale):
        """
        Returns:
            Tensor (N, 1): containing ln |det J| where J is the jacobian
        """
        return torch.log(scale).view(scale.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return 'dim={d}, swap={swap}'.format(**self.__dict__)


class ChannelCouplingBlock(CouplingBlock):
    """Channel-wise coupling layer for images.
    """

    def __init__(self, dim, nnet, mask_type='channel0'):
        if mask_type == 'channel0':
            swap = False
        elif mask_type == 'channel1':
            swap = True
        else:
            raise ValueError('Unknown mask type.')
        super(ChannelCouplingBlock, self).__init__(dim, nnet, swap)
        self.mask_type = mask_type

    def extra_repr(self):
        return 'dim={d}, mask_type={mask_type}'.format(**self.__dict__)


class MaskedCouplingBlock(nn.Module):
    """Coupling layer for images implemented using masks.
    """

    def __init__(self, dim, nnet, mask_type='checkerboard0'):
        nn.Module.__init__(self)
        self.d = dim
        self.nnet = nnet
        self.mask_type = mask_type

    def func_s_t(self, x):
        f = self.nnet(x)
        s = torch.sigmoid(f[:, :self.d] + 2.)
        t = f[:, self.d:]
        return s, t

    def forward(self, x, logpx=None):
        # get mask
        b = mask_utils.get_mask(x, mask_type=self.mask_type)

        # masked forward
        x_a = b * x
        s, t = self.func_s_t(x_a)
        y = (x * s + t) * (1 - b) + x_a

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(s, b)

    def inverse(self, y, logpy=None):
        # get mask
        b = mask_utils.get_mask(y, mask_type=self.mask_type)

        # masked forward
        y_a = b * y
        s, t = self.func_s_t(y_a)
        x = y_a + (1 - b) * (y - t) / s

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(s, b)

    def _logdetgrad(self, s, mask):
        return torch.log(s).mul_(1 - mask).view(s.shape[0], -1).sum(1, keepdim=True)

    def extra_repr(self):
        return 'dim={d}, mask_type={mask_type}'.format(**self.__dict__)
