import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):

    def forward(self, x):
        return x


class FullSort(nn.Module):

    def forward(self, x):
        return torch.sort(x, 1)[0]


class MaxMin(nn.Module):

    def forward(self, x):
        b, d = x.shape
        max_vals = torch.max(x.view(b, d // 2, 2), 2)[0]
        min_vals = torch.min(x.view(b, d // 2, 2), 2)[0]
        return torch.cat([max_vals, min_vals], 1)


class LipschitzCube(nn.Module):

    def forward(self, x):
        return (x >= 1).to(x) * (x - 2 / 3) + (x <= -1).to(x) * (x + 2 / 3) + ((x > -1) * (x < 1)).to(x) * x**3 / 3


class SwishFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, beta):
        beta_sigm = torch.sigmoid(beta * x)
        output = x * beta_sigm
        ctx.save_for_backward(x, output, beta)
        return output / 1.1

    @staticmethod
    def backward(ctx, grad_output):
        x, output, beta = ctx.saved_tensors
        beta_sigm = output / x
        grad_x = grad_output * (beta * output + beta_sigm * (1 - beta * output))
        grad_beta = torch.sum(grad_output * (x * output - output * output)).expand_as(beta)
        return grad_x / 1.1, grad_beta / 1.1


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


if __name__ == '__main__':

    m = Swish()
    xx = torch.linspace(-5, 5, 1000).requires_grad_(True)
    yy = m(xx)
    dd, dbeta = torch.autograd.grad(yy.sum() * 2, [xx, m.beta])

    import matplotlib.pyplot as plt

    plt.plot(xx.detach().numpy(), yy.detach().numpy(), label='Func')
    plt.plot(xx.detach().numpy(), dd.detach().numpy(), label='Deriv')
    plt.plot(xx.detach().numpy(), torch.max(dd.detach().abs() - 1, torch.zeros_like(dd)).numpy(), label='|Deriv| > 1')
    plt.legend()
    plt.tight_layout()
    plt.show()
