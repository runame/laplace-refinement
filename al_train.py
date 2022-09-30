import argparse
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
from torch.nn import functional as F

from models import autoguide, network
from models import pyro as pyro_models
from utils import data_utils, metrics, utils

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['map', 'hmc', 'refine', 'refine_sub'], default='map')
parser.add_argument('--dataset', choices=['fmnist', 'mnist'], default='fmnist')
parser.add_argument('--n_burnins', type=int, default=100)
parser.add_argument('--n_samples', type=int, default=200)
parser.add_argument('--chain_id', type=int, choices=[1, 2, 3, 4, 5], default=1)
parser.add_argument('--n_flows', type=int, default=5, help='Only relevant for refine and refine_sub method')
parser.add_argument('--flow_type', choices=['radial', 'planar'], default='radial', help='Only relevant for refine and refine_sub method')
parser.add_argument('--subspace_dim', type=int, default=100, help='Only relevant for refine_sub method')
parser.add_argument('--prior_precision', type=float, default=30)
parser.add_argument('--randseed', type=int, default=1)
args = parser.parse_args()


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.randseed)
np.random.seed(args.randseed)
torch.backends.cudnn.benchmark = True

# Random seeds for HMC initialization
chain2randseed = {1: 77, 2: 777, 3: 7777, 4: 77777, 5: 777777}

data_path = './data'
if not os.path.exists(data_path):
    os.makedirs(data_path)
# For saving pretrained models
save_path = f'./pretrained_models/{args.dataset}/al'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# The network
get_net = lambda: network.MLP(n_features=784, n_hiddens=50)

# Dataset --- no data augmentation
loader_fn = data_utils.get_mnist_loaders if args.dataset == 'mnist' else data_utils.get_fmnist_loaders
train_loader, val_loader, test_Loader = loader_fn(
    data_path, train=True, batch_size=128, model_class='MLP', download=True, device=DEVICE)

X_train, y_train = [], []
for x, y in train_loader:
    X_train.append(x); y_train.append(y)
X_train, y_train = torch.cat(X_train, 0), torch.cat(y_train, 0)

M, N = X_train.shape[0], math.prod(X_train.shape[1:])
print(f'[Randseed: {args.randseed}] Dataset: {args.dataset.upper()}, n_data: {M}, n_feat: {N}, n_param: {utils.count_params(get_net())}')

if args.method == 'map':
    model = get_net().cuda()
    wd = args.prior_precision / M  # Since we use averaged NLL
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    print(f'Weight decay: {wd}')

    schd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100*len(train_loader))

    pbar = tqdm.trange(100)
    for it in pbar:
        model.train()
        epoch_loss = 0

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            schd.step()
            opt.zero_grad()
            epoch_loss += loss.item()

        model.eval()
        train_acc = metrics.accuracy(*utils.predict(model, train_loader))
        pbar.set_description(f'[Loss: {epoch_loss:.3f}; Train acc: {train_acc:.1f}]')

    torch.save(model.state_dict(), f'{save_path}/{args.method}_{args.randseed}.pt')

elif args.method == 'refine':
    net = get_net()
    net.load_state_dict(torch.load(f'{save_path}/map_{args.randseed}.pt'))
    net.cuda()
    net.eval()

    la = Laplace(
        net, 'classification', subset_of_weights='all',
        hessian_structure='diag', prior_precision=args.prior_precision
    )
    la.fit(train_loader)
    la.optimize_prior_precision(method='CV', val_loader=val_loader, pred_type='nn', n_samples=10)

    base_dist_mean = la.mean
    base_dist_var = la.posterior_variance

    model = pyro_models.ClassificationModel(
        get_net, n_data=M, prior_prec=args.prior_precision, cuda=True
    )
    guide = autoguide.AutoNormalizingFlowCustom(
        model.model, base_dist_mean, base_dist_var, diag=True,
        flow_type=args.flow_type, flow_len=args.n_flows, cuda=True
    )

    n_epochs = 20
    n_iters = n_epochs * len(train_loader)
    schd = pyro.optim.CosineAnnealingLR({
        'optimizer': torch.optim.Adam, 'optim_args': {'lr': 1e-3, 'weight_decay': 0}, 'T_max': n_iters
    })

    svi = SVI(model.model, guide, optim=schd, loss=Trace_ELBO())
    pbar = tqdm.trange(n_epochs)

    for it in pbar:
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            loss = svi.step(x, y)
            schd.step()

        pbar.set_description(f'[Loss: {loss:.3f}]')

    state_dict = {
        'base_dist_mean': base_dist_mean, 'base_dist_var': base_dist_var,
        'flow_type': args.flow_type, 'flow_len': args.n_flows,
        'state_dict': guide.state_dict()
    }
    torch.save(state_dict, f'{save_path}/{args.method}_{args.flow_type}_{args.n_flows}_{args.randseed}.pt')

elif args.method == 'refine_sub':
    model = get_net()
    model.load_state_dict(torch.load(f'{save_path}/map_{args.randseed}.pt'))
    model.cuda()

    la = Laplace(
        model, 'classification', subset_of_weights='all',
        hessian_structure='diag', prior_precision=args.prior_precision
    )
    la.fit(train_loader)

    base_dist_mean = la.mean
    base_dist_scale = torch.sqrt(la.posterior_variance)
    base_dist = dist.Normal(base_dist_mean, base_dist_scale)

    # Projection matrix
    A = torch.zeros(len(base_dist_mean), args.subspace_dim, device=DEVICE)
    eigval_idxs = torch.argsort(la.posterior_variance, descending=True)[:args.subspace_dim]
    A[eigval_idxs, range(args.subspace_dim)] = 1
    A = A.T

    model = pyro_models.ClassificationModel(
        get_net, n_data=M, prior_prec=args.prior_precision, cuda=True,
        proj_mat=A, base_dist=base_dist, diag=True
    )
    guide = autoguide.AutoNormalizingFlowCustom(
        model.model_subspace, model.base_mean_proj, model.base_Cov_proj, diag=False,
        flow_type=args.flow_type, flow_len=args.n_flows, cuda=True
    )

    n_epochs = 20
    n_iters = n_epochs * len(train_loader)
    schd = pyro.optim.CosineAnnealingLR({
        'optimizer': torch.optim.Adam, 'optim_args': {'lr': 1e-1, 'weight_decay': 0}, 'T_max': n_iters
    })

    svi = SVI(model.model_subspace, guide, optim=schd, loss=Trace_ELBO())
    pbar = tqdm.trange(n_epochs)

    for it in pbar:
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            loss = svi.step(x, y)
            schd.step()

        pbar.set_description(f'[Loss: {loss:.3f}]')

    state_dict = {
        'proj_mat': A, 'base_dist_mean': model.base_mean_proj, 'base_dist_Cov': model.base_Cov_proj,
        'flow_type': args.flow_type, 'flow_len': args.n_flows,
        'state_dict': guide.state_dict()
    }
    torch.save(state_dict, f'{save_path}/{args.method}_{args.subspace_dim}_{args.flow_type}_{args.n_flows}_{args.randseed}.pt')

elif args.method == 'hmc':
    model = pyro_models.ClassificationModel(
        get_net=get_net, n_data=M, prior_prec=args.prior_precision, cuda=True
    )

    pyro.util.set_rng_seed(chain2randseed[args.chain_id])
    # Initialize the HMC by sampling from the prior
    mcmc_kernel = NUTS(model.model, init_strategy=init.init_to_sample)
    mcmc = MCMC(
        mcmc_kernel, num_samples=args.n_samples, warmup_steps=args.n_burnins,
    )

    mcmc.run(X_train.cuda(), y_train.cuda(), full_batch=True)

    # HMC samples
    hmc_samples = mcmc.get_samples()['theta'].flatten(1).cpu().numpy()
    np.save(f'{save_path}/{args.method}_{args.randseed}_{args.chain_id}.npy', hmc_samples)
