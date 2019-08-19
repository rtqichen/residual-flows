import argparse
import math
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets

from lib.iresnet import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', type=str, default='cifar10', choices=[
        'mnist',
        'cifar10',
        'celeba',
        'celebahq',
        'celeba_5bit',
        'imagenet32',
        'imagenet64',
    ]
)
parser.add_argument('--dataroot', type=str, default='data')
parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

# Sampling parameters.
parser.add_argument('--real', type=eval, choices=[True, False], default=False)
parser.add_argument('--nrow', type=int, default=10)
parser.add_argument('--ncol', type=int, default=10)
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--nbatches', type=int, default=5)
parser.add_argument('--save-each', type=eval, choices=[True, False], default=False)

parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=512)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=int, default=128)
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)

parser.add_argument('--task', type=str, choices=['density'], default='density')
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

parser.add_argument('--resume', type=str, required=True)
parser.add_argument('--nworkers', type=int, default=4)
args = parser.parse_args()

W = args.ncol
H = args.nrow

args.batchsize = W * H
args.val_batchsize = W * H

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    print('WARNING: Using device {}'.format(device))


def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_noise(x, nvals=255):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * nvals + noise
    x = x / (nvals + 1)
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_padding(x):
    if args.padding > 0:
        u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
        logpu = torch.zeros_like(u).sum([1, 2, 3])
        return torch.cat([u, x], dim=1), logpu
    else:
        return x, torch.zeros(x.shape[0]).to(x)


def remove_padding(x):
    if args.padding > 0:
        return x[:, args.padding:, :, :]
    else:
        return x


def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def update_lipschitz(model):
    for m in model.modules():
        if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
            m.compute_weight(update=True)
        if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
            m.compute_weight(update=True)


print('Loading dataset {}'.format(args.data), flush=True)
# Dataset and hyperparameters
if args.data == 'cifar10':
    im_dim = 3
    n_classes = 10

    if args.task in ['classification', 'hybrid']:

        if args.real:

            # Classification-specific preprocessing.
            transform_train = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                add_noise,
            ])

            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])

        # Remove the logit transform.
        init_layer = layers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        if args.real:
            transform_train = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                add_noise,
            ])
            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise,
            ])
        init_layer = layers.LogitTransform(0.05)
    if args.real:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataroot, train=True, transform=transform_train),
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.nworkers,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataroot, train=False, transform=transform_test),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
elif args.data == 'mnist':
    im_dim = 1
    init_layer = layers.LogitTransform(1e-6)
    n_classes = 10

    if args.real:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.dataroot, train=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor(),
                    add_noise,
                ])
            ),
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.nworkers,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.dataroot, train=False, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor(),
                    add_noise,
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
elif args.data == 'svhn':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    n_classes = 10

    if args.real:
        train_loader = torch.utils.data.DataLoader(
            vdsets.SVHN(
                args.dataroot, split='train', download=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
                    transforms.ToTensor(),
                    add_noise,
                ])
            ),
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.nworkers,
        )
        test_loader = torch.utils.data.DataLoader(
            vdsets.SVHN(
                args.dataroot, split='test', download=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor(),
                    add_noise,
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.nworkers,
        )
elif args.data == 'celebahq':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)

    if args.real:
        train_loader = torch.utils.data.DataLoader(
            datasets.CelebAHQ(
                train=True, transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    reduce_bits,
                    lambda x: add_noise(x, nvals=2**args.nbits),
                ])
            ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CelebAHQ(
                train=False, transform=transforms.Compose([
                    reduce_bits,
                    lambda x: add_noise(x, nvals=2**args.nbits),
                ])
            ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
elif args.data == 'celeba_5bit':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        print('Changing image size to 64.')
        args.imagesize = 64

    if args.real:
        train_loader = torch.utils.data.DataLoader(
            datasets.CelebA5bit(
                train=True, transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    lambda x: add_noise(x, nvals=32),
                ])
            ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CelebA5bit(train=False, transform=transforms.Compose([
                lambda x: add_noise(x, nvals=32),
            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
elif args.data == 'imagenet32':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 32:
        print('Changing image size to 32.')
        args.imagesize = 32

    if args.real:
        train_loader = torch.utils.data.DataLoader(
            datasets.Imagenet32(train=True, transform=transforms.Compose([
                add_noise,
            ])), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.Imagenet32(train=False, transform=transforms.Compose([
                add_noise,
            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )
elif args.data == 'imagenet64':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        print('Changing image size to 64.')
        args.imagesize = 64

    if args.real:
        train_loader = torch.utils.data.DataLoader(
            datasets.Imagenet64(train=True, transform=transforms.Compose([
                add_noise,
            ])), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.Imagenet64(train=False, transform=transforms.Compose([
                add_noise,
            ])), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers
        )

if args.task in ['classification', 'hybrid']:
    try:
        n_classes
    except NameError:
        raise ValueError('Cannot perform classification with {}'.format(args.data))
else:
    n_classes = 1

print('Dataset loaded.', flush=True)
print('Creating model.', flush=True)

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)

# Model
model = ResidualFlow(
    input_size,
    n_blocks=list(map(int, args.nblocks.split('-'))),
    intermediate_dim=args.idim,
    factor_out=args.factor_out,
    quadratic=args.quadratic,
    init_layer=init_layer,
    actnorm=args.actnorm,
    fc_actnorm=args.fc_actnorm,
    batchnorm=args.batchnorm,
    dropout=args.dropout,
    fc=args.fc,
    coeff=args.coeff,
    vnorms=args.vnorms,
    n_lipschitz_iters=args.n_lipschitz_iters,
    sn_atol=args.sn_tol,
    sn_rtol=args.sn_tol,
    n_power_series=args.n_power_series,
    n_dist=args.n_dist,
    n_samples=args.n_samples,
    kernels=args.kernels,
    activation_fn=args.act,
    fc_end=args.fc_end,
    fc_idim=args.fc_idim,
    n_exact_terms=args.n_exact_terms,
    preact=args.preact,
    neumann_grad=args.neumann_grad,
    grad_in_forward=args.mem_eff,
    first_resblock=args.first_resblock,
    learn_p=args.learn_p,
    block_type=args.block,
)

model.to(device)

print('Initializing model.', flush=True)

with torch.no_grad():
    x = torch.rand(1, *input_size[1:]).to(device)
    model(x)
print('Restoring from checkpoint.', flush=True)
checkpt = torch.load(args.resume)
state = model.state_dict()
model.load_state_dict(checkpt['state_dict'], strict=True)

ema = utils.ExponentialMovingAverage(model)
ema.set(checkpt['ema'])
ema.swap()

print(model, flush=True)

model.eval()
print('Updating lipschitz.', flush=True)
update_lipschitz(model)


def visualize(model):
    utils.makedirs('{}_imgs_t{}'.format(args.data, args.temp))

    with torch.no_grad():

        for i in tqdm(range(args.nbatches)):
            # random samples
            rand_z = torch.randn(args.batchsize, (im_dim + args.padding) * args.imagesize * args.imagesize).to(device)
            rand_z = rand_z * args.temp
            fake_imgs = model(rand_z, inverse=True).view(-1, *input_size[1:])
            if args.squeeze_first: fake_imgs = squeeze_layer.inverse(fake_imgs)
            fake_imgs = remove_padding(fake_imgs)
            fake_imgs = fake_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
            fake_imgs = fake_imgs.cpu()

            if args.save_each:
                for j in range(fake_imgs.shape[0]):
                    save_image(
                        fake_imgs[j], '{}_imgs_t{}/{}.png'.format(args.data, args.temp, args.batchsize * i + j), nrow=1,
                        padding=0, range=(0, 1), pad_value=0
                    )
            else:
                save_image(
                    fake_imgs, 'imgs/{}_t{}_samples{}.png'.format(args.data, args.temp, i), nrow=W, padding=2,
                    range=(0, 1), pad_value=1
                )


real_imgs = test_loader.__iter__().__next__()[0] if args.real else None
if args.real:
    real_imgs = test_loader.__iter__().__next__()[0]
    save_image(
        real_imgs.cpu().float(), 'imgs/{}_real.png'.format(args.data), nrow=W, padding=2, range=(0, 1), pad_value=1
    )

visualize(model)
