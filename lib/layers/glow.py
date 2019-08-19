import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleLinear(nn.Module):

    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward(self, x, logpx=None):
        y = F.linear(x, self.weight)
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad

    def inverse(self, y, logpy=None):
        x = F.linear(y, self.weight.inverse())
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight)))

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class InvertibleConv2d(nn.Module):

    def __init__(self, dim):
        super(InvertibleConv2d, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward(self, x, logpx=None):
        y = F.conv2d(x, self.weight.view(self.dim, self.dim, 1, 1))
        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad.expand_as(logpx) * x.shape[2] * x.shape[3]

    def inverse(self, y, logpy=None):
        x = F.conv2d(y, self.weight.inverse().view(self.dim, self.dim, 1, 1))
        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad.expand_as(logpy) * x.shape[2] * x.shape[3]

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight)))

    def extra_repr(self):
        return 'dim={}'.format(self.dim)
