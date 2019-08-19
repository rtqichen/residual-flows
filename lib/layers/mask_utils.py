import torch


def _get_checkerboard_mask(x, swap=False):
    n, c, h, w = x.size()

    H = ((h - 1) // 2 + 1) * 2  # H = h + 1 if h is odd and h if h is even
    W = ((w - 1) // 2 + 1) * 2

    # construct checkerboard mask
    if not swap:
        mask = torch.Tensor([[1, 0], [0, 1]]).repeat(H // 2, W // 2)
    else:
        mask = torch.Tensor([[0, 1], [1, 0]]).repeat(H // 2, W // 2)
    mask = mask[:h, :w]
    mask = mask.contiguous().view(1, 1, h, w).expand(n, c, h, w).type_as(x.data)

    return mask


def _get_channel_mask(x, swap=False):
    n, c, h, w = x.size()
    assert (c % 2 == 0)

    # construct channel-wise mask
    mask = torch.zeros(x.size())
    if not swap:
        mask[:, :c // 2] = 1
    else:
        mask[:, c // 2:] = 1
    return mask


def get_mask(x, mask_type=None):
    if mask_type is None:
        return torch.zeros(x.size()).to(x)
    elif mask_type == 'channel0':
        return _get_channel_mask(x, swap=False)
    elif mask_type == 'channel1':
        return _get_channel_mask(x, swap=True)
    elif mask_type == 'checkerboard0':
        return _get_checkerboard_mask(x, swap=False)
    elif mask_type == 'checkerboard1':
        return _get_checkerboard_mask(x, swap=True)
    else:
        raise ValueError('Unknown mask type {}'.format(mask_type))
