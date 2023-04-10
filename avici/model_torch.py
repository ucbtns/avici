import inspect
from inspect import signature
import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.distributions import Bernoulli

import functools
import numpy as np

from avici.utils.data_torch import torch_get_train_x, torch_get_x 

def layer_norm(shape):
    return nn.LayerNorm(normalized_shape=shape, elementwise_affine=True)

# def set_diagonal(arr, val):
#     n_vars = arr.shape[-1]
#     return arr.transpose(-1, -2).masked_fill(torch.eye(n_vars, dtype=torch.bool), val).transpose(-1, -2)

def set_diagonal(tensor, value):
        """
        Set the diagonal elements of a tensor to a given value.
        Args:
            tensor: Input tensor of shape (..., N, N)
            value: Value to set the diagonal elements to
        Returns:
            Tensor with modified diagonal elements
        """
        diag_mask = torch.eye(tensor.size(-1), dtype=torch.bool, device=tensor.device).expand_as(tensor)
        return tensor.masked_fill_(diag_mask, value)

class BaseModel(nn.Module):

    def __init__(self,
                 layers=8,
                 dim=128,
                 key_size=32,
                 num_heads=8,
                 widening_factor=4,
                 dropout=0.1,
                 out_dim=None,
                 logit_bias_init=-3.0,
                 cosine_temp_init=0.0,
                 ln_axis=-1,
                 name="BaseModel",
                 ):
        
        super(BaseModel, self).__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.cosine_temp_init = cosine_temp_init
        # self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")  # kaiming uniform/
        self.w_init = functools.partial(init.kaiming_uniform_, a=2.0, mode='fan_in', nonlinearity='relu')

    def forward(self, x, is_training: bool):
        
        dropout_rate = self.dropout if is_training else 0.0
        z = nn.Linear(self.dim, self.dim)(x)

        for _ in range(self.layers):
            # mha
            import pdb; pdb.set_trace()
            q_in = layer_norm(self.dim)(z) # query
            k_in = layer_norm(self.dim)(z) # key 
            v_in = layer_norm(self.dim)(z) # value
            z_attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads, kdim=self.key_size)(q_in, k_in, v_in)
            z = z + F.dropout(z_attn, dropout_rate)

            # ffn
            z_in = layer_norm(self.dim)(z)
            # This doesn;t have the right initialisation and dimensions will cause an issue
            z_ffn = nn.Sequential(
                    nn.Linear(self.widening_factor * self.dim, self.dim), 
                    nn.ReLU(),
                    nn.Linear(self.dim, self.dim)
                )(z_in)
            z = z + F.dropout(z_ffn, dropout_rate)

            # flip N and d axes
            z = torch.swapaxes(z, -3, -2)

        z = layer_norm(self.dim)(z)
        assert z.shape[-2] == x.shape[-2] and z.shape[-3] == x.shape[-3], "Do we have an odd number of layers?"

        # [..., n_vars, dim]
        z = torch.max(z, dim=-3)

        # u, v dibs embeddings for edge probabilities
         # This doesn;t have the right initialisation and dimensions will cause an issue
        u = nn.Sequential([
            layer_norm(axis=self.dim),
            nn.Linear(self.dim, self.out_dim),
        ])(z)
        v = nn.Sequential([
            layer_norm(self.dim),
            nn.Linear(self.dim, self.out_dim, bias=True),
        ])(z)

        # edge logits
        # [..., n_vars, dim], [..., n_vars, dim] -> [..., n_vars, n_vars]
        u /= torch.linalg.norm(u, dim=-1, ord=2, keepdim=True)
        v /= torch.linalg.norm(v, dim=-1, ord=2, keepdim=True)
        logit_ij = torch.einsum("...id,...jd->...ij", u, v)
        temp = torch.nn.Parameter(torch.zeros(1, 1, 1), requires_grad=True)
        init.constant_(temp, self.cosine_temp_init)
        temp = temp.squeeze()
        logit_ij *= torch.exp(temp)
        logit_ij_bias = nn.Parameter(torch.Tensor([1, 1, 1]))
        init.constant_(logit_ij_bias, self.logit_bias_init)
        logit_ij_bias = logit_ij_bias.squeeze()
        logit_ij += logit_ij_bias

        assert logit_ij.shape[-1] == x.shape[-2] and logit_ij.shape[-2] == x.shape[-2]
        return logit_ij
    

class InferenceModel(nn.Module):

    def __init__(self, *,
                 model_class,
                 model_kwargs,
                 train_p_obs_only=0.0,
                 acyclicity=None,
                 acyclicity_pow_iters=10,
                 mask_diag=True,
                 ):

        super(InferenceModel, self).__init__()
        self._train_p_obs_only = torch.tensor(train_p_obs_only)
        self._acyclicity_weight = acyclicity
        self._acyclicity_power_iters = acyclicity_pow_iters
        self.mask_diag = mask_diag

        # filter deprecated network kwargs
        sig = signature(model_class.__init__).parameters
        deprec = list(filter(lambda key: key not in sig, model_kwargs.keys()))
        for k in deprec:
            del model_kwargs[k]
            # print(f"Ignoring deprecated kwarg `{k}` loaded from `model_kwargs` in checkpoint")

        # init forward pass transform
        self.net = model_class(**model_kwargs)

    def forward(self, *args):
        # Pass the input arguments to the model and return the result.
        return self.net(*args)

    def sample_graphs(self, n_samples, params, rng, x):
        """
        Args:
            n_samples: number of samples
            params: torch.nn.Module.state_dict()
            rng: torch.Generator
            x: [..., N, d, 2]
            is_count_data [...] bool
        Returns:
            graph samples of shape [..., n_samples, d, d]
        """
        # [..., d, d]
        is_training = False
        logits = self.net(x, is_training)
        prob1 = torch.sigmoid(logits)

        # sample graphs
        # [..., n_samples, d, d]
        samples = Bernoulli(prob1).sample((n_samples,), generator=rng).permute(1, 0, 2, 3).type(torch.int32)
        if self.mask_diag:
            samples = set_diagonal(samples, 0.0)

        return samples

    def infer_edge_logprobs(self, params, rng, x, is_training: bool):
        """
        Args:
            params: torch.nn.Module.state_dict()
            rng: torch.Generator
            x: [..., N, d, 2]
            is_training
            is_count_data [...] bool
        Returns:
            logprobs of graph adjacency matrix prediction of shape [..., d, d]
        """
        # [..., d, d]
        logits = self.net(x, is_training)
        logp_edges = torch.nn.LogSigmoid(logits)
        if self.mask_diag:
            logp_edges = self.set_diagonal(logp_edges, float('-inf'))

        return logp_edges
    
    def infer_edge_probs(self, params, x):
        """
        For test time inference
        Args:
            params: torch.nn.Module.state_dict()
            x: [..., N, d, 1]
            is_count_data [...] bool
        Returns:
            probabilities of graph adjacency matrix prediction of shape [..., d, d]
        """
        is_training_, dummy_rng_ = False, torch.Generator()  # assume test time
        logp_edges = self.infer_edge_logprobs(params, dummy_rng_, x, is_training_)
        p_edges = torch.exp(logp_edges)
        return p_edges

    def exp_matmul(self, _logmat, _vec, _axis):
        """
        Matrix-vector multiplication with matrix in log-space
        """
        if _axis == -1:
            _ax_unsqueeze, _ax_sum = -2, -1
        elif _axis == -2:
            _ax_unsqueeze, _ax_sum = -1, -2
        else:
            raise ValueError(f"invalid axis inside exp_matmul")

        _logvec = torch.logsumexp(_logmat + _vec.unsqueeze(_ax_unsqueeze), dim=_ax_sum)
        return torch.exp(_logvec)

    def acyclicity_spectral_log(self, logmat, rng, power_iterations):
        """
        No Bears acyclicity constraint by
        https://psb.stanford.edu/psb-online/proceedings/psb20/Lee.pdf
        Performed in log-space
        """

        # left/right power iteration
        u = torch.randn(logmat.shape[:-1], generator=rng, device=logmat.device)
        v = torch.randn(logmat.shape[:-1], generator=rng, device=logmat.device)

        for t in range(power_iterations):
            # u_new = torch.einsum('...i,...ij->...j', u, mat)
            u_new = self.exp_matmul(logmat, u, -2)  # u @ exp(mat)

            # v_new = torch.einsum('...ij,...j->...i', mat, v)
            v_new = self.exp_matmul(logmat, v, -1)  # exp(mat) @ v

            u = u_new / torch.linalg.norm(u_new, ord=2, dim=-1, keepdim=True)
            v = v_new / torch.linalg.norm(v_new, ord=2, dim=-1, keepdim=True)

        u = u.detach()
        v = v.detach()

        # largest_eigenvalue = (u @ exp(mat) @ v) / u.dot(v)
        largest_eigenvalue = (
                torch.einsum('...j,...j->...', u, self.exp_matmul(logmat, v, -1)) /
                torch.einsum('...j,...j->...', u, v)
        )

        return largest_eigenvalue

    def loss(self, params, dual, key, data, t, is_training: bool):
        # `data` leaves have leading dimension [1, batch_size_device, ...]

        key, subk = torch.split(key, 1)
        if is_training:
            x = get_train_x(subk, data, p_obs_only=self._train_p_obs_only)
        else:
            x = get_x(data)

        n_vars = data["g"].shape[-1]

        ### inference model q(G | D)
        # [..., n_observations, d, 2] --> [..., d, d]
        key, subk = torch.split(key, 1)
        logits = self.net(params, subk, x, is_training)

        # get logits [..., d, d]
        logp1 = torch.nn.LogSigmoid(  logits)
        logp0 = torch.nn.LogSigmoid(- logits)

        # labels [..., d, d]
        y_soft = data["g"]

        # mean over edges and skip diagonal (no self-loops)
        # [...]
        loss_eltwise = - (y_soft * logp1 + (1 - y_soft) * logp0)
        if self.mask_diag:
            batch_loss = set_diagonal(loss_eltwise, 0.0).sum((-1, -2)) / (n_vars * (n_vars - 1))
        else:
            batch_loss = loss_eltwise.sum((-1, -2)) / (n_vars * n_vars)

        # [] scalar
        loss_raw = batch_loss.mean() # mean over all available batch dims

        ### acyclicity
        key, subk = torch.split(key, 1)
        if self._acyclicity_weight is not None:
            # [..., d, d]
            if self.mask_diag:
                logp_edges = set_diagonal(logp1, float('-inf'))
            else:
                logp_edges = logp1

            # [...]
            spectral_radii = self.acyclicity_spectral_log(logp_edges, subk, power_iterations=self._acyclicity_power_iters)

            # [] scalars
            ave_acyc_penalty = spectral_radii.mean()
            wgt_acyc_penalty = self._acyclicity_weight(ave_acyc_penalty, t, dual)

        else:
            # [] scalars
            ave_acyc_penalty = torch.tensor(0.0, device=logp1.device)
            wgt_acyc_penalty = torch.tensor(0.0, device=logp1.device)

        # [] scalar
        loss = loss_raw + wgt_acyc_penalty
        aux = {
            "loss_raw": loss_raw,
            "acyc": ave_acyc_penalty,
            "wgt_acyc": wgt_acyc_penalty,
            "mean_z_norm": torch.abs(logits).mean(),
        }
        return loss, aux
