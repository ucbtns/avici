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
from avici.utils.multihead import MultiHeadAttention

def layer_norm(shape, dtype = torch.float32):
    return nn.LayerNorm(normalized_shape=shape, elementwise_affine=True, dtype=dtype)

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

linear_1_mapping = { "1": 0, "3": 1, "5": 2, "7": 3, "9": 4, "11": 5,"13": 6, "15": 7,"17": 8, "19": 9,"21": 10, "23": 11, "25": 12,"27": 13, "29": 14,  "31": 15,}
linear_1_list = list(linear_1_mapping.keys())
linear_2_mapping = {"2": 0,"4": 1,"6": 2, "8": 3, "10": 4,"12": 5,"14": 6,"16": 7,"18": 8,"20": 9, "22": 10,"24": 11, "26": 12, "28": 13,"30": 14,"32": 15,}
linear_2_list = list(linear_2_mapping.keys())

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
                 dtype=torch.float32, 
                 grad = False,
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
        self.w_init = functools.partial(init.kaiming_uniform_, a=2.0, mode='fan_in', nonlinearity='relu')
        self.dtype= dtype
        self.grad = grad

        # Initialisation
        self.linear_layer_1 = nn.Linear(2, self.dim, dtype=self.dtype) 

        # Inside the attention loop:
        self.layer_norms = nn.ModuleList([layer_norm(self.dim, dtype=self.dtype) for _ in range(self.layers * (3+1))])
        self.attentions = nn.ModuleList([MultiHeadAttention(num_heads=self.num_heads,
                                                            key_size=self.key_size, 
                                                            w_init_scale=2.0,
                                                            model_size=self.dim,
                                                            dtype=self.dtype)for _ in range(self.layers)])
        self.linear_layers_1 = nn.ModuleList([nn.Linear(self.dim, self.widening_factor * self.dim, dtype=self.dtype) for _ in range(self.layers)])
        [self.w_init(self.linear_layers_1[i].weight) for i in range(self.layers)]
        self.linear_layers_2 = nn.ModuleList([nn.Linear(self.widening_factor * self.dim, self.dim, dtype=self.dtype) for _ in range(self.layers)])
        [self.w_init(self.linear_layers_2[i].weight) for i in range(self.layers)]

        self.layer_norm_1 = layer_norm(self.dim, dtype=self.dtype)
        
        # Edge Probs - u,v
        self.layer_norm_u = layer_norm(self.dim, dtype=self.dtype)
        self.linear_u = nn.Linear(self.dim, self.dim, dtype=self.dtype)
        self.w_init(self.linear_u.weight)

        self.layer_norm_v = layer_norm(self.dim, dtype=self.dtype)
        self.linear_v = nn.Linear(self.dim, self.dim, dtype=self.dtype)
        self.w_init(self.linear_v.weight)

        # Logit mult:
        self.temp = nn.Parameter(torch.zeros(1, 1, 1, dtype=self.dtype), requires_grad=True)
        init.constant_(self.temp, self.cosine_temp_init)
        
        self.logit_ij_bias = nn.Parameter(torch.Tensor([1]).to(self.dtype))
        init.constant_(self.logit_ij_bias, self.logit_bias_init)
    
    def set_params(self, params):
        linear_count, linear_1_count, linear_2_count = 0,0,0

        with torch.no_grad():
            for key, value in params.items():
                try:   
                    split_key = key.split('/')
                    layers = split_key[1].split('_')
                    if layers[0] == 'layer':
                        if len(layers) == 2: nid = 0
                        else: nid = int(layers[-1])  
                        if nid < 64:
                            self.layer_norms[nid].bias.copy_(torch.tensor(params[key]['offset'], dtype =self.dtype))
                            self.layer_norms[nid].weight.copy_(torch.tensor(params[key]['scale'], dtype =self.dtype))
                        elif nid == 64:
                            self.layer_norm_1.bias.copy_(torch.tensor(params[key]['offset'], dtype =self.dtype))
                            self.layer_norm_1.weight.copy_(torch.tensor(params[key]['scale'], dtype =self.dtype))
                        elif nid == 65:
                            self.layer_norm_u.bias.copy_(torch.tensor(params[key]['offset'], dtype =self.dtype))
                            self.layer_norm_u.weight.copy_(torch.tensor(params[key]['scale'], dtype =self.dtype))
                        elif nid == 66:
                            self.layer_norm_v.bias.copy_(torch.tensor(params[key]['offset'], dtype =self.dtype))
                            self.layer_norm_v.weight.copy_(torch.tensor(params[key]['scale'], dtype =self.dtype))
                    elif layers[0] == 'linear':
                        if linear_count == 0:
                            self.linear_layer_1.weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.linear_layer_1.bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                        elif layers[-1] in linear_1_list:
                            self.linear_layers_1[linear_1_mapping[layers[-1]]].weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.linear_layers_1[linear_1_mapping[layers[-1]]].bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                            linear_1_count += 1
                        elif layers[-1] in linear_2_list:
                            self.linear_layers_2[linear_2_mapping[layers[-1]]].weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.linear_layers_2[linear_2_mapping[layers[-1]]].bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                            linear_2_count += 1
                        elif layers[-1] == '33':
                            self.linear_u.weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.linear_u.bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                        elif layers[-1] == '34':
                            self.linear_v.weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.linear_v.bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                        linear_count += 1
                    elif layers[0] == 'multi':
                        if len(layers) == 3: nid = 0
                        else: nid = int(layers[-1]) 
                        if split_key[-1] == 'key':
                            self.attentions[nid].key.weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.attentions[nid].key.bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                        elif split_key[-1] == 'query':
                            self.attentions[nid].query.weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.attentions[nid].query.bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                        elif split_key[-1] == 'value':
                            self.attentions[nid].value.weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.attentions[nid].value.bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                        elif split_key[-1] == 'linear':
                            self.attentions[nid].linear.weight.copy_(torch.tensor(params[key]['w'], dtype =self.dtype).T)
                            self.attentions[nid].linear.bias.copy_(torch.tensor(params[key]['b'], dtype =self.dtype))
                except: 
                    self.temp = torch.nn.Parameter(torch.tensor(params[key]['learned_temp'], dtype=self.dtype),requires_grad=False)
                    self.logit_ij_bias = torch.nn.Parameter(torch.tensor(params[key]['final_matrix_bias'], dtype =self.dtype),requires_grad=False)
            
        return self.children()
                          
    def forward(self, x, is_training: bool):
        dropout_rate = self.dropout if is_training else 0.0
        z = self.linear_layer_1(x) # [n,d,2] --> [n, d, dim]

        layer_norm_idx = 0
        for i in range(self.layers):
            # mha
            q_in = self.layer_norms[layer_norm_idx](z) # query --> [n, d, dim]
            k_in = self.layer_norms[layer_norm_idx + 1](z) # key --> [n, d, dim]
            v_in = self.layer_norms[layer_norm_idx + 2](z) # value --> [n, d, dim]
            layer_norm_idx += 3

            z_attn = self.attentions[i](q_in, k_in, v_in) # [n, d, dim]
            if is_training:
                z = z + F.dropout(z_attn, dropout_rate) # [n, d, dim]
            else:
                z = z + z_attn

            # ffn
            z_in = self.layer_norms[layer_norm_idx](z)# [n, d, dim]
            layer_norm_idx += 1
            z_ffn_1 = nn.ReLU()(self.linear_layers_1[i](z_in))
            z_ffn = self.linear_layers_2[i](z_ffn_1)
            if is_training:
                z = z + F.dropout(z_ffn, dropout_rate)
            else:
                z = z + z_ffn

            # flip N and d axes
            z = torch.swapaxes(z, -3, -2)
        
        z = self.layer_norm_1(z)
        assert z.shape[-2] == x.shape[-2] and z.shape[-3] == x.shape[-3], "Do we have an odd number of layers?"

        # [..., n_vars, dim]
        z,_ = torch.max(z, dim=-3)

        # u, v dibs embeddings for edge probabilities
        u_norm =  self.layer_norm_u(z)# [n, d, dim]
        u_edge = self.linear_u(u_norm)
        
        v_norm =  self.layer_norm_v(z)# [n, d, dim]
        v_edge = self.linear_v(v_norm)

        # edge logits
        # [..., n_vars, dim], [..., n_vars, dim] -> [..., n_vars, n_vars]
        u_edge /= torch.linalg.norm(u_edge, dim=-1, ord=2, keepdim=True)
        v_edge /= torch.linalg.norm(v_edge, dim=-1, ord=2, keepdim=True)
        logit_ij = torch.einsum("...id,...jd->...ij", u_edge, v_edge)
        temp = self.temp.squeeze()
        logit_ij *= torch.exp(temp)
        logit_ij_bias = self.logit_ij_bias.squeeze()
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
                 params=None,
                 ):

        super(InferenceModel, self).__init__()
        self._train_p_obs_only = torch.tensor(train_p_obs_only)
        self._acyclicity_weight = acyclicity
        self._acyclicity_power_iters = acyclicity_pow_iters
        self.mask_diag = mask_diag
        self.sigmoid = nn.LogSigmoid()

        # filter deprecated network kwargs
        sig = signature(model_class.__init__).parameters
        deprec = list(filter(lambda key: key not in sig, model_kwargs.keys()))
        for k in deprec:
            del model_kwargs[k]
            # print(f"Ignoring deprecated kwarg `{k}` loaded from `model_kwargs` in checkpoint")
        # init forward pass transform
        self.net = model_class(**model_kwargs)
        if params:
            child = self.net.set_params(params)
            self.net.children = child
        
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
        logp_edges = self.sigmoid(logits)

        if self.mask_diag:
            logp_edges = set_diagonal(logp_edges, float('-inf'))

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

        if is_training: x = torch_get_train_x(data, p_obs_only=self._train_p_obs_only)
        else: x = torch_get_x(data)

        n_vars = data["g"].shape[-1]

        ### inference model q(G | D)
        # [..., n_observations, d, 2] --> [..., d, d]
        self.net.children = self.net.set_params(params)
        logits = self.net(x, is_training)

        # get logits [..., d, d]
        logp1 = self.sigmoid(  logits)
        logp0 = self.sigmoid(- logits)

        # labels [..., d, d]
        y_soft = data["g"]

        # mean over edges and skip diagonal (no self-loops)
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
            if self.mask_diag: logp_edges = set_diagonal(logp1, float('-inf'))
            else: logp_edges = logp1

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
