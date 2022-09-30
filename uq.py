import argparse
import os
import warnings

import numpy as np
import pycalib.calibration_methods as calib
import pyro
import pyro.distributions as dist
import torch
import yaml
from laplace import Laplace
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

import utils.data_utils as du
import utils.utils as util
import utils.wilds_utils as wu
from baselines.swag.swag import fit_swag_and_precompute_bn_params
from models import network
from models import pyro as pyro_models
from utils import metrics as metrics_fns
from utils.test import test

warnings.filterwarnings('ignore')


def main(args):
    # set device and random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.prior_precision = util.get_prior_precision(args, device)
    util.set_seed(args.seed)

    # load in-distribution data
    in_data_loaders, ids, no_loss_acc = du.get_in_distribution_data_loaders(
        args, device)
    train_loader, val_loader, in_test_loader = in_data_loaders

    # fit models
    mixture_components, samples = fit_models(
        args, train_loader, val_loader, device)

    # evaluate models
    metrics = evaluate_models(
        args, mixture_components, in_test_loader, ids, no_loss_acc, samples, device)

    # save results
    util.save_results(args, metrics)


def fit_models(args, train_loader, val_loader, device):
    """ load pre-trained weights, fit inference methods, and tune hyperparameters """
    n_classes = 100 if args.benchmark == 'CIFAR-100-OOD' else 10

    mixture_components = list()
    all_samples = list()
    for model_idx in range(args.nr_components):
        samples = None
        if args.method in ['map', 'ensemble', 'laplace', 'mola', 'swag', 'multi-swag', 'bbb', 'csghmc']:
            model = util.load_pretrained_model(args, model_idx, n_classes, device)

        # For saving pretrained models
        dataset = 'fmnist' if args.benchmark in ['R-FMNIST', 'FMNIST-OOD'] else 'cifar10'
        if args.benchmark == 'CIFAR-100-OOD':
            dataset += '0'
        is_ll = args.subset_of_weights == 'last_layer' and args.method not in ['map', 'ensemble', 'bbb']
        layers = 'll' if is_ll else 'al'
        save_path = os.path.join(args.models_root, f'{dataset}/{layers}')
        if args.compute_mmd and model_idx == 0:
            CHAIN_IDS = [1, 2, 3]
            # Reference HMC samples for computing MMD distances
            hmc_samples = np.concatenate([
                np.load(f'{save_path}/hmc_{args.model_seed}_{cid}.npy')
                for cid in CHAIN_IDS])
            n_hmc_samples = len(hmc_samples)
            if args.nr_components > 1:
                if n_hmc_samples % args.nr_components != 0:
                    raise ValueError('n_hmc_samples must be divisible by nr_components.')
                n_hmc_samples = int(n_hmc_samples / args.nr_components)
            all_samples.append(hmc_samples)

        if args.method == 'map':
            if args.compute_mmd:
                if is_ll:
                    sample_model = model.fc if 'cifar' in dataset else model.ll
                else:
                    sample_model = model
                samples = torch.nn.utils.parameters_to_vector(
                    sample_model.parameters()).detach().repeat(n_hmc_samples, 1)
            if args.likelihood == 'classification' and args.use_temperature_scaling:
                print('Fitting temperature scaling model on validation data...')
                all_y_prob = [model(d[0].to(device)).detach().cpu() for d in val_loader]
                all_y_prob = torch.cat(all_y_prob, dim=0)
                all_y_true = torch.cat([d[1] for d in val_loader], dim=0)

                temperature_scaling_model = calib.TemperatureScaling()
                temperature_scaling_model.fit(all_y_prob.numpy(), all_y_true.numpy())
                model = (model, temperature_scaling_model)

        elif args.method in ['laplace', 'mola']:
            if type(args.prior_precision) is str: # file path
                prior_precision = torch.load(args.prior_precision, map_location=device)
            elif type(args.prior_precision) is float:
                prior_precision = args.prior_precision
            else:
                raise ValueError('prior precision has to be either float or string (file path)')
            Backend = util.get_backend(args.backend, args.approx_type)
            optional_args = dict()

            if args.subset_of_weights == 'last_layer':
                optional_args['last_layer_name'] = args.last_layer_name

            print('Fitting Laplace approximation...')

            model = Laplace(model, args.likelihood,
                            subset_of_weights=args.subset_of_weights,
                            hessian_structure=args.hessian_structure,
                            prior_precision=prior_precision,
                            temperature=args.temperature,
                            backend=Backend, **optional_args)
            model.fit(train_loader)

            if (args.optimize_prior_precision is not None) and (args.method == 'laplace'):
                if (type(prior_precision) is float) and (args.prior_structure != 'scalar'):
                    n = model.n_params if args.prior_structure == 'all' else model.n_layers
                    prior_precision = prior_precision * torch.ones(n, device=device)

                print('Optimizing prior precision for Laplace approximation...')

                verbose_prior = args.prior_structure == 'scalar'
                model.optimize_prior_precision(
                    method=args.optimize_prior_precision,
                    init_prior_prec=prior_precision,
                    val_loader=val_loader,
                    pred_type=args.pred_type,
                    link_approx=args.link_approx,
                    n_samples=args.n_samples,
                    verbose=verbose_prior
                )

            if args.compute_mmd:
                if args.hessian_structure == 'diag':
                    samples = dist.Normal(
                        model.mean, model.posterior_scale).sample((n_hmc_samples,))
                else:
                    samples = dist.MultivariateNormal(
                        model.mean, scale_tril=model.posterior_scale).sample((n_hmc_samples,))

        elif args.method in ['swag', 'multi-swag']:
            print('Fitting SWAG...')

            model = fit_swag_and_precompute_bn_params(
                model, device, train_loader, args.swag_n_snapshots,
                args.swag_lr, args.swag_c_epochs, args.swag_c_batches,
                args.data_parallel, args.n_samples, args.swag_bn_update_subset)

        elif args.method == 'bbb' and args.compute_mmd:
            posterior_mean = list()
            posterior_scale = list()
            for module in model.modules():
                if hasattr(module, 'mu_weight'):
                    posterior_mean.append(module.mu_weight)
                    posterior_scale.append(module.rho_weight)
                if hasattr(module, 'mu_kernel'):
                    posterior_mean.append(module.mu_kernel)
                    posterior_scale.append(module.rho_kernel)
                if hasattr(module, 'mu_bias'):
                    posterior_mean.append(module.mu_bias)
                    posterior_scale.append(module.rho_bias)
            posterior_mean = torch.nn.utils.parameters_to_vector(posterior_mean)
            posterior_scale = torch.log1p(torch.exp(
                torch.nn.utils.parameters_to_vector(posterior_scale)))
            samples = dist.Normal(
                posterior_mean, posterior_scale).sample((n_hmc_samples,))

        elif args.method not in ['ensemble', 'bbb', 'csghmc']:
            if args.method == 'hmc' and not args.compute_mmd:
                raise ValueError(f'compute_mmd needs to be set to True for {args.method}.')

            # The network
            if is_ll:
                get_net = (lambda: network.WideResNet(16, 4, num_classes=n_classes) 
                           if 'CIFAR' in args.benchmark else network.LeNet())
            else:
                get_net = lambda: network.MLP(784, n_hiddens=50)

            X_train = torch.cat([x for x, _ in train_loader], dim=0)
            y_train = torch.cat([y for _, y in train_loader], dim=0)
            M, N = X_train.shape[0], np.prod(X_train.shape[1:])
            if model_idx == 0:
                print(f'[Randseed: {args.model_seed}] Dataset: {args.benchmark}, '
                    f'n_data: {M}, n_feat: {N}, n_param: {util.count_params(get_net())}')
                print()

            net = get_net()
            model_seed = args.model_seed
            if args.nr_components > 1:
                model_seed = model_idx + 1
            net.load_state_dict(torch.load(f'{save_path}/map_{model_seed}.pt', map_location=device))
            net.to(device)
            net.eval()

            cuda = torch.cuda.is_available()
            subset_of_weights = 'last_layer' if is_ll else 'all'
            hessian_structure = (
                'full' if is_ll and args.benchmark != 'CIFAR-100-OOD'
                else 'diag'
            )
            N_FEAT = 256 if 'CIFAR-10' in args.benchmark else 84
            if args.n_samples % args.nr_components != 0:
                raise ValueError('n_samples must be divisible by nr_components.')
            num_samples = int(args.n_samples / args.nr_components)

            if is_ll:
                model = pyro_models.ClassificationModelLL(
                    n_data=M, n_classes=n_classes, n_features=N_FEAT,
                    feature_extractor=net.forward_features,
                    prior_prec=args.prior_precision, cuda=cuda)
            else:
                model = pyro_models.ClassificationModel(
                    get_net, n_data=M, prior_prec=args.prior_precision, cuda=cuda)

            if args.method == 'vb':
                pyro.get_param_store().clear()
                guide = AutoDiagonalNormal(model.model)
                guide._setup_prototype(X_train[:2].to(device), y_train[:2].to(device))
                guide.load_state_dict(torch.load(f'{save_path}/{args.method}_{model_seed}.pt'))
                samples = guide.get_posterior().sample((n_hmc_samples,))
                predictive = Predictive(model.model, guide=guide, num_samples=num_samples, return_sites=('_RETURN',))

            elif 'nf_naive' in args.method:
                pyro.get_param_store().clear()
                state_dict = torch.load(f'{save_path}/{args.method}_{model_seed}.pt')

                guide = util.load_nf_guide(model.model, state_dict, *next(iter(train_loader)), cuda=True, method='nf_naive')
                samples = guide.get_posterior().sample((n_hmc_samples,))
                predictive = Predictive(model.model, guide=guide, num_samples=num_samples, return_sites=('_RETURN',))

            elif 'refine' in args.method and 'sub' not in args.method:
                pyro.get_param_store().clear()
                state_dict = torch.load(f'{save_path}/{args.method}_{model_seed}.pt')

                la = Laplace(
                    net, 'classification', subset_of_weights=subset_of_weights,
                    hessian_structure=hessian_structure, prior_precision=args.prior_precision)
                la.fit(train_loader)
                la.optimize_prior_precision()

                diag = hessian_structure == 'diag'
                guide = util.load_nf_guide(model.model, state_dict, *next(iter(train_loader)), diag=diag, cuda=cuda)
                samples = guide.get_posterior().sample((n_hmc_samples,))
                predictive = Predictive(model.model, guide=guide, num_samples=num_samples, return_sites=('_RETURN',))

            elif args.method == 'refine_sub':
                pyro.get_param_store().clear()
                state_dict = torch.load(f'{save_path}/refine_sub_{model_seed}.pt')

                la = Laplace(
                    net, 'classification', subset_of_weights='last_layer',
                    hessian_structure='full', prior_precision=args.prior_precision)
                la.fit(train_loader)
                la.optimize_prior_precision()
                base_dist = dist.MultivariateNormal(la.mean, la.posterior_covariance)

                model = pyro_models.ClassificationModelLL(
                    n_data=M, n_classes=n_classes, n_features=N_FEAT,
                    feature_extractor=net.forward_features,
                    prior_prec=args.prior_precision, cuda=cuda,
                    proj_mat=state_dict['proj_mat'], base_dist=base_dist)

                guide = util.load_nf_guide(model.model_subspace, state_dict, *next(iter(train_loader)), cuda=cuda)
                samples = guide.get_posterior().sample((n_hmc_samples,)) @ state_dict['proj_mat']
                predictive = Predictive(model.model_subspace, guide=guide, num_samples=num_samples, return_sites=('_RETURN',))

            else:  # 'hmc'
                samples = torch.as_tensor(hmc_samples, dtype=torch.float, device=device)
                predictive = Predictive(model.model, {'theta': samples.to(device)}, return_sites=('_RETURN',))

            model = (predictive, net)

        if args.likelihood == 'regression' and args.sigma_noise is None:
            print('Optimizing noise standard deviation on validation data...')
            args.sigma_noise = wu.optimize_noise_standard_deviation(model, val_loader, device)

        mixture_components.append(model)
        all_samples.append(samples)

    if len(all_samples) > 2:
        method_samples = torch.cat(all_samples[1:])
        assert n_hmc_samples * args.nr_components == len(method_samples)
        all_samples = [all_samples[0], method_samples]

    return mixture_components, all_samples


def evaluate_models(args, mixture_components, in_test_loader, ids, no_loss_acc, samples, device):
    """ evaluate the models and return relevant evaluation metrics """

    metrics = []
    for i, id in enumerate(ids):
        # load test data
        test_loader = in_test_loader if i == 0 else du.get_ood_test_loader(
            args, id)
        
        use_no_loss_acc = no_loss_acc if i > 0 else False
        # make model predictions and compute some metrics
        test_output, test_time = util.timing(lambda: test(
            mixture_components, test_loader, args.method,
            pred_type=args.pred_type, link_approx=args.link_approx,
            n_samples=args.n_samples, device=device, no_loss_acc=use_no_loss_acc,
            likelihood=args.likelihood, sigma_noise=args.sigma_noise))
        some_metrics, all_y_prob, all_y_var = test_output
        some_metrics['test_time'] = test_time

        if i == 0:
            all_y_prob_in = all_y_prob.clone()

        # compute more metrics, aggregate and print them:
        # log likelihood, accuracy, confidence, Brier sore, ECE, MCE, AUROC, FPR95
        more_metrics = compute_metrics(
            i, id, all_y_prob, test_loader, all_y_prob_in, all_y_var, samples, args)
        metrics.append({**some_metrics, **more_metrics})
        print(', '.join([f'{k}: {v:.4f}' for k, v in metrics[-1].items()]))

    return metrics


def compute_metrics(i, id, all_y_prob, test_loader, all_y_prob_in, all_y_var, samples, args):
    """ compute evaluation metrics """

    metrics = {}

    # compute Brier, ECE and MCE for in-distribution and distribution shift/WILDS data
    if i == 0 or args.benchmark in ['R-MNIST', 'R-FMNIST', 'CIFAR-10-C', 'ImageNet-C']:
        if args.benchmark in ['R-MNIST', 'R-FMNIST', 'CIFAR-10-C', 'ImageNet-C']:
            print(f'{args.benchmark} with distribution shift intensity {i}')
        labels = torch.cat([data[1] for data in test_loader])
        metrics['brier'] = util.get_brier_score(all_y_prob, labels)
        metrics['ece'], metrics['mce'] = util.get_calib(all_y_prob, labels)

    # compute AUROC and FPR95 for OOD benchmarks
    if 'OOD' in args.benchmark:
        print(f'{args.benchmark} - dataset: {id}')
        if i > 0:
            # compute other metrics
            metrics['auroc'] = util.get_auroc(all_y_prob_in, all_y_prob)
            metrics['fpr95'], _ = util.get_fpr95(all_y_prob_in, all_y_prob)

    # compute regression calibration
    if args.benchmark == 'WILDS-poverty':
        print(f'{args.benchmark} with distribution shift intensity {i}')
        labels = torch.cat([data[1] for data in test_loader])
        metrics['calib_regression'] = util.get_calib_regression(
            all_y_prob.numpy(), all_y_var.sqrt().numpy(), labels.numpy())

    # compute MMD to HMC samples
    if args.compute_mmd and i == 0 and len(samples) == 2:
        hmc_samples, method_samples = samples[0], samples[1]
        if method_samples is not None:
            metrics['mmd_to_hmc'] = metrics_fns.mmd_rbf(
                hmc_samples, method_samples.cpu().numpy())

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str,
                        choices=['R-MNIST', 'R-FMNIST', 'CIFAR-10-C', 'ImageNet-C',
                                 'MNIST-OOD', 'FMNIST-OOD', 'CIFAR-10-OOD', 'CIFAR-100-OOD',
                                 'WILDS-camelyon17', 'WILDS-iwildcam',
                                 'WILDS-civilcomments', 'WILDS-amazon',
                                 'WILDS-fmow', 'WILDS-poverty'],
                        default='CIFAR-10-C', help='name of benchmark')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--download', action='store_true',
                        help='if True, downloads the datasets needed for given benchmark')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                    help='fraction of data to use (only supported for WILDS)')
    parser.add_argument('--models_root', type=str, default='./models',
                        help='root of pre-trained models')
    parser.add_argument('--model_seed', type=int, default=None,
                        help='random seed with which model(s) were trained')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--hessians_root', type=str, default='./hessians',
                        help='root of pre-computed Hessians')
    parser.add_argument('--method', type=str, default='laplace',
                        help='name of method to use')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--compute_mmd', action='store_true',
                        help='Compute MMD to HMC samples.')

    parser.add_argument('--pred_type', type=str,
                        choices=['nn', 'glm'],
                        default='glm',
                        help='type of approximation of predictive distribution')
    parser.add_argument('--link_approx', type=str,
                        choices=['mc', 'probit', 'bridge'],
                        default='probit',
                        help='type of approximation of link function')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='nr. of MC samples for approximating the predictive distribution')

    parser.add_argument('--likelihood', type=str, choices=['classification', 'regression'],
                        default='classification', help='likelihood for Laplace')
    parser.add_argument('--subset_of_weights', type=str, choices=['last_layer', 'all'],
                        default='last_layer', help='subset of weights for Laplace')
    parser.add_argument('--backend', type=str, choices=['backpack', 'kazuki'], default='backpack')
    parser.add_argument('--approx_type', type=str, choices=['ggn', 'ef'], default='ggn')
    parser.add_argument('--hessian_structure', type=str, choices=['diag', 'kron', 'full'],
                        default='kron', help='structure of the Hessian approximation')
    parser.add_argument('--last_layer_name', type=str, default=None,
                        help='name of the last layer of the model')
    parser.add_argument('--prior_precision', type=float, default=1.,
                        help='prior precision to use for computing the covariance matrix')
    parser.add_argument('--optimize_prior_precision', default=None,
                        choices=['marglik', 'CV'],
                        help='optimize prior precision according to specified method')
    parser.add_argument('--prior_structure', type=str, default='scalar',
                        choices=['scalar', 'layerwise', 'all'])
    parser.add_argument('--sigma_noise', type=float, default=None,
                        help='noise standard deviation for regression (if -1, optimize it)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the likelihood.')

    parser.add_argument('--swag_n_snapshots', type=int, default=40,
                        help='number of snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_batches', type=int, default=None,
                        help='number of batches between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_epochs', type=int, default=1,
                        help='number of epochs between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_lr', type=float, default=1e-2,
                        help='learning rate for [Multi]SWAG')
    parser.add_argument('--swag_bn_update_subset', type=float, default=1.0,
                        help='fraction of train data for updating the BatchNorm statistics for [Multi]SWAG')

    parser.add_argument('--nr_components', type=int, default=1,
                        help='number of mixture components to use')
    parser.add_argument('--mixture_weights', type=str,
                        choices=['uniform', 'optimize'],
                        default='uniform',
                        help='how the mixture weights for MoLA are chosen')

    parser.add_argument('--model', type=str, default='WRN16-4',
                        choices=['MLP', 'FMNIST-MLP', 'LeNet', 'WRN16-4', 'WRN16-4-fixup', 'WRN50-2',
                                 'LeNet-BBB-reparam', 'LeNet-BBB-flipout', 'LeNet-CSGHMC',
                                 'WRN16-4-BBB-reparam', 'WRN16-4-BBB-flipout', 'WRN16-4-CSGHMC'],
                         help='the neural network model architecture')
    parser.add_argument('--no_dropout', action='store_true', help='only for WRN-fixup.')
    parser.add_argument('--data_parallel', action='store_true',
                        help='if True, use torch.nn.DataParallel(model)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for testing')
    parser.add_argument('--val_set_size', type=int, default=2000,
                        help='size of validation set (taken from test set)')
    parser.add_argument('--use_temperature_scaling', default=False,
                        help='if True, calibrate model using temperature scaling')

    parser.add_argument('--job_id', type=int, default=0,
                        help='job ID, leave at 0 when running locally')
    parser.add_argument('--config', default=None, nargs='+',
                        help='YAML config file path')
    parser.add_argument('--run_name', type=str, help='overwrite save file name')
    parser.add_argument('--noda', action='store_true')
    parser.add_argument('--normalize', action='store_true')

    args = parser.parse_args()
    args_dict = vars(args)

    # load config file (YAML)
    if args.config is not None:
        for path in args.config:
            with open(path) as f:
                config = yaml.full_load(f)
            args_dict.update(config)

    if args.data_parallel and (args.method in ['laplace, mola']):
        raise NotImplementedError(
            'laplace and mola do not support DataParallel yet.')

    if (args.optimize_prior_precision is not None) and (args.method == 'mola'):
        raise NotImplementedError(
            'optimizing the prior precision for MoLA is not supported yet.')

    if args.mixture_weights != 'uniform':
        raise NotImplementedError(
            'Only uniform mixture weights are supported for now.')

    if ((args.method in ['ensemble', 'mola', 'multi-swag'])
        and (args.nr_components <= 1)):
        parser.error(
            'Choose nr_components > 1 for ensemble, MoLA, or MultiSWAG.')

    if args.model != 'WRN16-4-fixup' and args.no_dropout:
        parser.error(
            'No dropout option only available for Fixup.')

    if args.benchmark in ['R-MNIST', 'MNIST-OOD', 'R-FMNIST', 'FMNIST-OOD']:
        if 'LeNet' not in args.model and 'MLP' not in args.model:
            parser.error('Only LeNet or (FMNIST-)MLP works for (F-)MNIST.')
    elif args.benchmark in ['CIFAR-10-C', 'CIFAR-10-OOD']:
        if 'WRN16-4' not in args.model:
            parser.error('Only WRN16-4 works for CIFAR-10-C.')
    elif args.benchmark == 'ImageNet-C':
        if not (args.model == 'WRN50-2'):
            parser.error('Only WRN50-2 works for ImageNet-C.')

    if args.benchmark == 'WILDS-poverty':
        args.likelihood = 'regression'
    else:
        args.likelihood = 'classification'

    for key, val in args_dict.items():
        print(f'{key}: {val}')
    print()

    main(args)
