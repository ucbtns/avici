import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, axis, eps=1e-5):
        super().__init__()
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, x.shape[self.axis:], eps=self.eps)


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
        super().__init__()
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

        self.linear = nn.Linear(self.dim, self.dim)
        self.layer_norms = nn.ModuleList([LayerNorm(axis=ln_axis) for _ in range(self.layers * 2 + 2)])
        self.attentions = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.dim,
                                                               num_heads=self.num_heads,
                                                               kdim=self.key_size,
                                                               vdim=self.dim) for _ in range(self.layers)])
        self.linears1 = nn.ModuleList([nn.Linear(self.dim, self.widening_factor * self.dim) for _ in range(self.layers)])
        self.linears2 = nn.ModuleList([nn.Linear(self.widening_factor * self.dim, self.dim) for _ in range(self.layers)])
        self.linear_u = nn.Linear(self.dim, self.out_dim)
        self.linear_v = nn.Linear(self.dim, self.out_dim)

        self.learned_temp = nn.Parameter(torch.tensor(cosine_temp_init))
        self.final_matrix_bias = nn.Parameter(torch.tensor(logit_bias_init))

    def forward(self, x, is_training: bool):
        dropout_rate = self.dropout if is_training else 0.0
        z = self.linear(x)

        layer_norm_idx = 0

        for i in range(self.layers):
            # mha
            q_in = self.layer_norms[layer_norm_idx](z)
            k_in = self.layer_norms[layer_norm_idx + 1](z)
            v_in = self.layer_norms[layer_norm_idx + 2](z)
            layer_norm_idx += 3

            z_attn, _ = self.attentions[i](q_in, k_in, v_in)
            z_attn = F.dropout(z_attn, dropout_rate)
            z = z + z_attn

            # ffn
            z_in = self.layer_norms[layer_norm_idx](z)
            layer_norm_idx += 1
            z_ffn = F.relu(self.linears1[i](z_in))
            z_ffn = F.dropout(self.linears2[i](z_ffn), dropout_rate)
            z = z + z_ffn

            # flip N and d axes
            z = torch.swapaxes(z, -3, -2)

        z = self.layer_norms[layer_norm_idx](z)
        # Make sure z and x have the same shape along the last two dimensions
        assert z.shape[-2] == x.shape[-2] and z.shape[-3] == x.shape[-3], "Do we have an odd number of layers?"

        # [..., n_vars, dim]
        z = torch.max(z, dim=-3)

        # u, v dibs embeddings for edge probabilities
        u = nn.Sequential(
            nn.LayerNorm(self.ln_axis),
            nn.Linear(self.out_dim, bias=True))(z)
        v = nn.Sequential(
            nn.LayerNorm(self.ln_axis),
            nn.Linear(self.out_dim, bias=True))(z)

        # edge logits
        # [..., n_vars, dim], [..., n_vars, dim] -> [..., n_vars, n_vars]
        u /= torch.linalg.norm(u, dim=-1, ord=2, keepdim=True)
        v /= torch.linalg.norm(v, dim=-1, ord=2, keepdim=True)
        logit_ij = torch.einsum("...id,...jd->...ij", u, v)
        temp = nn.Parameter(torch.tensor(self.cosine_temp_init).unsqueeze(0).unsqueeze(0).unsqueeze(0))
        logit_ij *= torch.exp(temp)
        logit_ij_bias = nn.Parameter(torch.tensor(self.logit_bias_init).unsqueeze(0).unsqueeze(0).unsqueeze(0))
        logit_ij += logit_ij_bias

        assert logit_ij.shape[-1] == x.shape[-2] and logit_ij.shape[-2] == x.shape[-2]
        return logit_ij



class InferenceModel:

    def __init__(self, *,
                 model_class,
                 model_kwargs,
                 train_p_obs_only=0.0,
                 acyclicity=None,
                 acyclicity_pow_iters=10,
                 mask_diag=True,
                 ):

        self._train_p_obs_only = torch.tensor(train_p_obs_only)
        self._acyclicity_weight = acyclicity
        self._acyclicity_power_iters = acyclicity_pow_iters
        self.mask_diag = mask_diag

        # filter deprecated network kwargs
        sig = inspect.signature(model_class.__init__).parameters
        deprec = list(filter(lambda key: key not in sig, model_kwargs.keys()))
        for k in deprec:
            del model_kwargs[k]
            # print(f"Ignoring deprecated kwarg `{k}` loaded from `model_kwargs` in checkpoint")

        # init forward pass transform
        self.net = model_class(**model_kwargs)

    def sample_graphs(self, n_samples, params, rng, x):
        """
        Args:
            n_samples: number of samples
            params: hk.Params
            rng
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
        samples = torch.bernoulli(prob1.unsqueeze(-3).expand(n_samples, *prob1.shape)).transpose(0, -3).type(torch.int32)
        if self.mask_diag:
            samples = set_diagonal(samples, 0.0)

        return samples

    def infer_edge_logprobs(self, params, rng, x, is_training: bool):
        """
        Args:
            params: hk.Params
            rng
            x: [..., N, d, 2]
            is_training
            is_count_data [...] bool
        Returns:
            logprobs of graph adjacency matrix prediction of shape [..., d, d]
        """
        # [..., d, d]
        logits = self.net(x, is_training)
        logp_edges = F.logsigmoid(logits)
        if self.mask_diag:
            logp_edges = set_diagonal(logp_edges, -float('inf'))

        return logp_edges

    def infer_edge_probs(self, params, x):
        """
        For test time inference
        Args:
            params: hk.Params
            x: [..., N, d, 1]
            is_count_data [...] bool
        Returns:
            probabilities of graph adjacency matrix prediction of shape [..., d, d]
        """
        is_training_ = False  # assume test time
        logp_edges = self.infer_edge_logprobs(params, 0, x, is_training_)
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

        _logvec, _logvec_sgn = logsumexp(_logmat, b=jnp.expand_dims(_vec, axis=_ax_unsqueeze),
                                         axis=_ax_sum, return_sign=True)
        return _logvec_sgn * jnp.exp(_logvec)


    def acyclicity_spectral_log(self, logmat, key, power_iterations):
        """
        No Bears acyclicity constraint by
        https://psb.stanford.edu/psb-online/proceedings/psb20/Lee.pdf

        Performed in log-space
        """

        # left/right power iteration
        key, subk = random.split(key)
        u = random.normal(subk, shape=logmat.shape[:-1])
        key, subk = random.split(key)
        v = random.normal(subk, shape=logmat.shape[:-1])

        for t in range(power_iterations):
            # u_new = jnp.einsum('...i,...ij->...j', u, mat)
            u_new = self.exp_matmul(logmat, u, -2) # u @ exp(mat)

            # v_new = jnp.einsum('...ij,...j->...i', mat, v)
            v_new = self.exp_matmul(logmat, v, -1) # exp(mat) @ v

            u = u_new / jnp.linalg.norm(u_new, ord=2, axis=-1, keepdims=True)
            v = v_new / jnp.linalg.norm(v_new, ord=2, axis=-1, keepdims=True)

        u = jax.lax.stop_gradient(u)
        v = jax.lax.stop_gradient(v)

        # largest_eigenvalue = (u @ exp(mat) @ v) / u.dot(v)
        largest_eigenvalue = (
                jnp.einsum('...j,...j->...', u, self.exp_matmul(logmat, v, -1)) /
                jnp.einsum('...j,...j->...', u, v)
        )

        return largest_eigenvalue


    """Training"""
    def loss(self, params, dual, key, data, t, is_training: bool):
        # `data` leaves have leading dimension [1, batch_size_device, ...]

        key, subk = random.split(key)
        if is_training:
            x = jax_get_train_x(subk, data, p_obs_only=self._train_p_obs_only)
        else:
            x = jax_get_x(data)

        n_vars = data["g"].shape[-1]

        ### inference model q(G | D)
        # [..., n_observations, d, 2] --> [..., d, d]
        key, subk = random.split(key)
        logits = self.net.apply(params, subk, x, is_training)

        # get logits [..., d, d]
        logp1 = jax.nn.log_sigmoid(  logits)
        logp0 = jax.nn.log_sigmoid(- logits)

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
        key, subk = random.split(key)
        if self._acyclicity_weight is not None:
            # [..., d, d]
            if self.mask_diag:
                logp_edges = set_diagonal(logp1, -jnp.inf)
            else:
                logp_edges = logp1

            # [...]
            spectral_radii = self.acyclicity_spectral_log(logp_edges, subk, power_iterations=self._acyclicity_power_iters)

            # [] scalars
            ave_acyc_penalty = spectral_radii.mean()
            wgt_acyc_penalty = self._acyclicity_weight(ave_acyc_penalty, t, dual)

        else:
            # [] scalars
            ave_acyc_penalty = jnp.array(0.0)
            wgt_acyc_penalty = jnp.array(0.0)

        # [] scalar
        loss = loss_raw + wgt_acyc_penalty
        aux = {
            "loss_raw": loss_raw,
            "acyc": ave_acyc_penalty,
            "wgt_acyc": wgt_acyc_penalty,
            "mean_z_norm": jnp.abs(logits).mean(),
        }
        return loss, aux
