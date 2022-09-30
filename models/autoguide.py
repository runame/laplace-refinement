import functools

import torch
from pyro import distributions as dist
from pyro.infer.autoguide import AutoContinuous, AutoNormalizingFlow
from pyro.infer.autoguide.initialization import init_to_feasible

FLOW_TYPES = {
    'planar': dist.transforms.planar,
    'radial': dist.transforms.radial
}


class AutoNormalizingFlowCustom(AutoNormalizingFlow):
    """
    AutoNormalizingFlow guide with a custom base distribution
    """

    def __init__(self, model, base_dist_mean, base_dist_Cov, diag=False, flow_type='radial', flow_len=5, cuda=False):
        init_transform_fn = functools.partial(dist.transforms.iterated, flow_len, FLOW_TYPES[flow_type])
        super().__init__(model, init_transform_fn)

        self.base_dist_mean = base_dist_mean
        self.base_dist_Cov = base_dist_Cov

        if diag:
            assert self.base_dist_Cov.shape == self.base_dist_mean.shape
            self.base_dist = dist.Normal(self.base_dist_mean, torch.sqrt(self.base_dist_Cov))
        else:
            self.base_dist = dist.MultivariateNormal(self.base_dist_mean, self.base_dist_Cov)

        self.transform = None
        self._prototype_tensor = torch.tensor(0.0, device='cuda' if cuda else 'cpu')
        self.cuda = cuda

    def get_base_dist(self):
        return self.base_dist

    def get_posterior(self, *args, **kwargs):
        if self.transform is None:
            self.transform = self._init_transform_fn(self.latent_dim)

            if self.cuda:
                self.transform.to('cuda:0')

            # Update prototype tensor in case transform parameters
            # device/dtype is not the same as default tensor type.
            for _, p in self.named_pyro_params():
                self._prototype_tensor = p
                break
        return super().get_posterior(*args, **kwargs)


class AutoNormalizingFlowCuda(AutoContinuous):

    def __init__(self, model, init_transform_fn, cuda=True):
        super().__init__(model, init_loc_fn=init_to_feasible)
        self._init_transform_fn = init_transform_fn
        self.transform = None
        self._prototype_tensor = torch.tensor(0.0, device='cuda' if cuda else 'cpu')
        self.cuda = cuda

    def get_base_dist(self):
        loc = self._prototype_tensor.new_zeros(1)
        scale = self._prototype_tensor.new_ones(1)
        return dist.Normal(loc, scale).expand([self.latent_dim]).to_event(1)


    def get_transform(self, *args, **kwargs):
        return self.transform


    def get_posterior(self, *args, **kwargs):
        if self.transform is None:
            self.transform = self._init_transform_fn(self.latent_dim)

            if self.cuda:
                self.transform.to('cuda:0')

            # Update prototype tensor in case transform parameters
            # device/dtype is not the same as default tensor type.
            for _, p in self.named_pyro_params():
                self._prototype_tensor = p
                break
        return super().get_posterior(*args, **kwargs)
