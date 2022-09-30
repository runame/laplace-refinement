import argparse
import functools
import math
import os

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import tqdm
from laplace import Laplace
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import *
from pyro.infer.autoguide import initialization as init
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import autoguide, network
from models import pyro as pyro_models
from utils import data_utils, metrics, utils

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['map', 'hmc', 'refine', 'nf_naive'], default='map')
parser.add_argument('--dataset', choices=['fmnist', 'cifar10', 'cifar100'], default='fmnist')
parser.add_argument('--n_burnins', type=int, default=100)
parser.add_argument('--n_samples', type=int, default=200)
parser.add_argument('--chain_id', type=int, choices=[1, 2, 3, 4, 5], default=1)
parser.add_argument('--n_flows', type=int, default=5, help='Only relevant for `args.method == "refine"`')
parser.add_argument('--flow_type', choices=['radial', 'planar'], default='radial', help='Only relevant for `args.method == "refine"`')
parser.add_argument('--prior_precision', type=float, default=None)
parser.add_argument('--randseed', type=int, default=1)
args = parser.parse_args()


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.randseed)
np.random.seed(args.randseed)
torch.backends.cudnn.benchmark = True

# Random seeds for HMC initialization
chain2randseed = {1: 77, 2: 777, 3: 7777, 4: 77777, 5: 777777}

# For saving pretrained models
save_path = f'./pretrained_models/{args.dataset}/ll'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Prior precision
if args.prior_precision is None:
    # From gridsearch
    args.prior_precision = 40 if 'cifar' in args.dataset else 510

print(f'Prior prec: {args.prior_precision}')

N_FEAT = 256 if args.dataset != 'fmnist' else 84
N_CLASS = 10 if args.dataset  != 'cifar100' else 100

# The network
get_net = lambda: network.WideResNet(16, 4, num_classes=N_CLASS) if 'cifar' in args.dataset else network.LeNet()

# Dataset --- no data augmentation
data_path = './data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

if args.dataset == 'fmnist':
    train_loader, val_loader, test_loader = data_utils.get_fmnist_loaders(data_path, download=True, device=DEVICE)
elif args.dataset == 'cifar10':
    train_loader, val_loader, test_loader = data_utils.get_cifar10_loaders(
        data_path, data_augmentation=(args.method == 'map'), normalize=False, download=True)
else:
    train_loader, val_loader, test_loader = data_utils.get_cifar100_loaders(
        data_path, data_augmentation=(args.method == 'map'), normalize=False, download=True)

X_train, y_train = [], []
for x, y in train_loader:
    X_train.append(x); y_train.append(y)
X_train, y_train = torch.cat(X_train, 0), torch.cat(y_train, 0)

M, N = X_train.shape[0], math.prod(X_train.shape[1:])
print(f'[Randseed: {args.randseed}] Dataset: {args.dataset.upper()}, n_data: {M}, n_feat: {N}, n_param: {utils.count_params(get_net())}')

if args.method == 'map':
    model = get_net().cuda()

    if 'cifar' in args.dataset:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    schd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100*len(train_loader))
    scaler = amp.GradScaler()

    pbar = tqdm.trange(100)
    for it in pbar:
        model.train()
        epoch_loss = 0

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            with amp.autocast():
                out = model(x)
                loss = F.cross_entropy(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            schd.step()
            opt.zero_grad()

            epoch_loss += loss.item()

        model.eval()
        train_acc = metrics.accuracy(*utils.predict(model, train_loader))
        pbar.set_description(f'[Loss: {epoch_loss:.3f}; Train acc: {train_acc:.1f}]')

    torch.save(model.state_dict(), f'{save_path}/{args.method}_{args.randseed}.pt')

elif args.method == 'nf_naive':
    net = get_net()
    net.load_state_dict(torch.load(f'{save_path}/map_{args.randseed}.pt'))
    net.cuda()
    net.eval()

    model = pyro_models.ClassificationModelLL(
        n_data=M, n_classes=N_CLASS, n_features=N_FEAT, feature_extractor=net.forward_features,
        prior_prec=args.prior_precision, cuda=True
    )

    nf = dist.transforms.planar if args.flow_type == 'planar' else dist.transforms.radial
    transform_init = functools.partial(dist.transforms.iterated, args.n_flows, nf)
    guide = autoguide.AutoNormalizingFlowCuda(model.model, transform_init, cuda=True)

    n_epochs = 20
    n_iters = n_epochs * len(train_loader)
    schd = pyro.optim.CosineAnnealingLR({
        'optimizer': torch.optim.Adam, 'optim_args': {'lr': 1e-3, 'weight_decay': 0}, 'T_max': n_iters
    })

    svi = SVI(model.model, guide, optim=schd, loss=Trace_ELBO())
    pbar = tqdm.trange(20)

    # Cache features for faster training
    features_train, y_train = utils.get_features(net, train_loader)
    train_loader_features = DataLoader(
        TensorDataset(features_train.cpu(), y_train.unsqueeze(-1).cpu()),
        batch_size=128, shuffle=True, num_workers=0
    )

    for it in pbar:
        for x, y in train_loader_features:
            x, y = x.cuda(), y.cuda()
            loss = svi.step(x, y, X_is_features=True)
            schd.step()

        pbar.set_description(f'[Loss: {loss:.3f}]')

    state_dict = {
        'flow_type': args.flow_type, 'flow_len': args.n_flows,
        'state_dict': guide.state_dict()
    }
    torch.save(state_dict, f'{save_path}/{args.method}_{args.flow_type}_{args.n_flows}_{args.randseed}.pt')


elif args.method == 'refine':
    net = get_net()
    net.load_state_dict(torch.load(f'{save_path}/map_{args.randseed}.pt'))
    net.cuda()
    net.eval()

    hess_str = 'diag' if args.dataset == 'cifar100' else 'full'

    la = Laplace(
        net, 'classification', subset_of_weights='last_layer',
        hessian_structure=hess_str, prior_precision=args.prior_precision
    )
    la.fit(train_loader)
    la.optimize_prior_precision()

    if hess_str == 'diag':
        base_dist_cov = la.posterior_variance
        base_dist = dist.Normal(la.mean, base_dist_cov.sqrt())
        diag = True
    else:
        base_dist_cov = la.posterior_covariance
        base_dist = dist.MultivariateNormal(la.mean, base_dist_cov)
        diag = False

    model = pyro_models.ClassificationModelLL(
        n_data=M, n_classes=N_CLASS, n_features=N_FEAT, feature_extractor=net.forward_features,
        prior_prec=args.prior_precision, cuda=True
    )

    guide = autoguide.AutoNormalizingFlowCustom(
        model.model, base_dist.mean, base_dist_cov, diag=diag,
        flow_type=args.flow_type, flow_len=args.n_flows, cuda=True
    )

    n_epochs = 20
    n_iters = n_epochs * len(train_loader)
    schd = pyro.optim.CosineAnnealingLR({
        'optimizer': torch.optim.Adam, 'optim_args': {'lr': 1e-3, 'weight_decay': 0}, 'T_max': n_iters
    })

    svi = SVI(model.model, guide, optim=schd, loss=Trace_ELBO())
    pbar = tqdm.trange(20)

    # Cache features for faster training
    features_train, y_train = utils.get_features(net, train_loader)
    train_loader_features = DataLoader(
        TensorDataset(features_train.cpu(), y_train.unsqueeze(-1).cpu()),
        batch_size=128, shuffle=True, num_workers=0
    )

    for it in pbar:
        for x, y in train_loader_features:
            x, y = x.cuda(), y.cuda()
            loss = svi.step(x, y, X_is_features=True)
            schd.step()

        pbar.set_description(f'[Loss: {loss:.3f}]')

    cov_key = 'base_dist_var' if diag else 'base_dist_Cov'
    state_dict = {
        'base_dist_mean': base_dist.mean, cov_key: base_dist_cov,
        'flow_type': args.flow_type, 'flow_len': args.n_flows,
        'state_dict': guide.state_dict()
    }

    torch.save(state_dict, f'{save_path}/{args.method}_{args.flow_type}_{args.n_flows}_{args.randseed}.pt')

elif args.method == 'hmc':
    net = get_net()
    net.load_state_dict(torch.load(f'{save_path}/map_{args.randseed}.pt'))
    net.cuda()
    net.eval()

    model = pyro_models.ClassificationModelLL(
        n_data=M, n_classes=N_CLASS, n_features=N_FEAT, feature_extractor=net.forward_features,
        prior_prec=args.prior_precision, cuda=True
    )

    pyro.util.set_rng_seed(chain2randseed[args.chain_id])
    # Initialize the HMC by sampling from the prior
    mcmc_kernel = NUTS(model.model, init_strategy=init.init_to_sample)
    mcmc = MCMC(
        mcmc_kernel, num_samples=args.n_samples, warmup_steps=args.n_burnins,
    )

    features_train, y_train = utils.get_features(net, train_loader)
    mcmc.run(features_train, y_train, full_batch=True, X_is_features=True)

    # HMC samples
    hmc_samples = mcmc.get_samples()['theta'].flatten(1).cpu().numpy()
    np.save(f'{save_path}/{args.method}_{args.randseed}_{args.chain_id}.npy', hmc_samples)
