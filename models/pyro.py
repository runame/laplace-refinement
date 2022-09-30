import math

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch.nn import functional as F

from utils import utils


class Model:

    def __init__(self, n_params, n_data, prior_prec=10, cuda=False,
                 proj_mat=None, base_dist=None, diag=False):
        self.uuid = np.random.randint(low=0, high=10000, size=1)[0]

        self.n_data = n_data
        self.prior_mean = torch.zeros(n_params, device='cuda' if cuda else 'cpu')
        self.prior_std = math.sqrt(1/prior_prec)
        self.cuda = cuda

        if proj_mat is not None:
            self.A = proj_mat
            self.k, self.d = self.A.shape

            self.prior_mean_proj = torch.zeros(self.k)
            self.prior_Cov_proj = self.prior_std**2 * self.A @ self.A.T

            if base_dist is not None:
                self.base_dist = base_dist

                self.base_mean_proj = self.A @ self.base_dist.mean

                if not diag:
                    self.base_Cov_proj = self.A @ self.base_dist.covariance_matrix @ self.A.T
                else:
                    self.base_Cov_proj = self.A * (self.base_dist.scale**2)[None, :] @ self.A.T

                self.base_dist_proj = dist.MultivariateNormal(self.base_mean_proj, self.base_Cov_proj)

        if cuda:
            self.prior_mean = self.prior_mean.cuda()

            if proj_mat is not None and base_dist is not None:
                self.prior_mean_proj = self.prior_mean_proj.cuda()
                self.prior_Cov_proj = self.prior_Cov_proj.cuda()

    def model(self, X, y=None):
        raise NotImplementedError()

    def model_subspace(self, X, y=None):
        raise NotImplementedError()


class RegressionModel(Model):

    def __init__(self, get_net, n_data, prior_prec=10, log_noise=torch.tensor(1.), cuda=False, proj_mat=None, base_dist=None):
        n_params = sum(p.numel() for p in self.get_net().parameters())
        super().__init__(n_params, n_data, prior_prec, cuda, proj_mat, base_dist)
        self.get_net = get_net
        self.noise = F.softplus(log_noise)

    def model(self, X, y=None):
        # Sample params from the prior
        theta = pyro.sample('theta', dist.Normal(self.prior_mean, self.prior_std).to_event(1))

        # Put the sample into the net
        net = self.get_net()
        utils.vector_to_parameters_backpropable(theta, net)
        f_X = net(X).squeeze()

        # Likelihood
        if y is not None:
            # Training
            with pyro.plate('data', size=self.n_data, subsample=y.squeeze()):
                pyro.sample('obs', dist.Normal(f_X, self.noise), obs=y.squeeze())
        else:
            # Testing
            pyro.sample('obs', dist.Normal(f_X, self.noise))

    def model_subspace(self, X, y=None, full_batch=False):
        # Sample params from the prior on low-dim, then project it to high-dim
        z = pyro.sample('z', dist.MultivariateNormal(self.prior_mean_proj, self.prior_Cov_proj))
        theta = self.A.T @ z

        # Put the sample into the net
        net = self.get_net()
        utils.vector_to_parameters_backpropable(theta, net)
        f_X = net(X).squeeze()

        # Likelihood
        if y is not None:
            # Training
            with pyro.plate('data', size=self.n_data, subsample=y.squeeze()):
                pyro.sample('obs', dist.Normal(f_X, self.noise), obs=y.squeeze())
        else:
            # Testing
            pyro.sample('obs', dist.Normal(f_X, self.noise))


class ClassificationModel(Model):

    def __init__(self, get_net, n_data, prior_prec=10, cuda=False, proj_mat=None, base_dist=None, diag=True):
        self.get_net = get_net

        n_params = sum(p.numel() for p in self.get_net().parameters())
        super().__init__(n_params, n_data, prior_prec, cuda, proj_mat, base_dist, diag)

    def model(self, X, y=None, full_batch=False):
        # Sample params from the prior
        theta = pyro.sample('theta', dist.Normal(self.prior_mean, self.prior_std).to_event(1))

        # Put the sample into the net
        net = self.get_net()

        if self.cuda:
            net.cuda()

        utils.vector_to_parameters_backpropable(theta, net)
        f_X = net(X)

        # Likelihood
        if y is not None:
            subsample = None if full_batch else y.squeeze()

            with pyro.plate('data', size=self.n_data, subsample=subsample):
                pyro.sample('obs', dist.Categorical(logits=f_X), obs=y.squeeze())

        return f_X

    def model_subspace(self, X, y=None, full_batch=False):
        # Sample params from the prior on low-dim, then project it to high-dim
        z = pyro.sample('z', dist.MultivariateNormal(self.prior_mean_proj, self.prior_Cov_proj))
        theta = self.A.T @ z

        # Put the sample into the net
        net = self.get_net()

        if self.cuda:
            net.cuda()

        utils.vector_to_parameters_backpropable(theta, net)
        f_X = net(X)

        # Likelihood
        if y is not None:
            subsample = None if full_batch else y.squeeze()

            with pyro.plate('data', size=self.n_data, subsample=subsample):
                pyro.sample('obs', dist.Categorical(logits=f_X), obs=y.squeeze())

        return f_X


class ClassificationModelLL(Model):

    def __init__(self, n_data, n_features, n_classes, feature_extractor, prior_prec=10, cuda=False, proj_mat=None, base_dist=None, diag=False):
        n_params = n_features*n_classes + n_classes  # weights and biases
        super().__init__(n_params, n_data, prior_prec, cuda, proj_mat, base_dist, diag)

        self.n_features = n_features
        self.n_classes = n_classes
        self.feature_extractor = feature_extractor

    def model(self, X, y=None, full_batch=False, X_is_features=False):
        # Sample params from the prior
        theta = pyro.sample('theta', dist.Normal(self.prior_mean, self.prior_std).to_event(1))
        f_X = self._forward(X, theta, X_is_features)

        # Likelihood
        if y is not None:
            subsample = None if full_batch else y.squeeze()

            with pyro.plate('data', size=self.n_data, subsample=subsample):
                pyro.sample('obs', dist.Categorical(logits=f_X), obs=y.squeeze())

        return f_X

    def _forward(self, X, theta, X_is_features=False):
        # Make it compatible with PyTorch's parameters vectorization that Laplace uses
        W = theta[:self.n_features*self.n_classes].reshape(self.n_classes, self.n_features)
        b = theta[self.n_classes]

        if X_is_features:
            phi_X = X
        else:
            with torch.no_grad():
                phi_X = self.feature_extractor(X)

        return phi_X @ W.T + b  # Transpose following nn.Linear
