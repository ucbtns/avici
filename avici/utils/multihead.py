"""(Multi-Head) Attention module for use in Transformer architectures."""

from typing import Optional
import warnings


import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def variance_scaling(scale, mode="fan_in"):
    def _initializer(tensor):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        
        if mode == "fan_in":
            factor = fan_in
        elif mode == "fan_out":
            factor = fan_out
        elif mode == "average":
            factor = (fan_in + fan_out) / 2
        else:
            raise ValueError("Invalid mode. Choose from 'fan_in', 'fan_out', or 'average'.")
        
        std = scale * torch.sqrt(torch.tensor(1.0 / factor))
        with torch.no_grad():
            return tensor.normal_(0, std.item())

    return _initializer

class MultiHeadAttention(nn.Module):
  """Multi-headed attention (MHA) module from HK.MultiHeadAttention.
  This module is intended for attending over sequences of vectors.
  Rough sketch:
  - Compute keys (K), queries (Q), and values (V) as projections of inputs.
  - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
  - Output is another projection of WV^T.
  For more detail, see the original Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762.
  Glossary of shapes:
  - T: Sequence length.
  - D: Vector (embedding) size.
  - H: Number of attention heads.
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      w_init_scale: float,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None,
  ):
    """Initialises the module.
    Args:
      num_heads: Number of independent attention heads (H).
      key_size: The size of keys (K) and queries used for attention.
      w_init_scale: DEPRECATED. Please use w_init instead.
      w_init: Initialiser for weights in the linear map. Once `w_init_scale` is
        fully deprecated `w_init` will become mandatory. Until then it has a
        default value of `None` for backwards compatability.
      value_size: Optional size of the value projection (V). If None, defaults
        to the key size (K).
      model_size: Optional size of the output embedding (D'). If None, defaults
        to the key size multiplied by the number of heads (K * H).
      name: Optional name for this module.
    """
    super().__init__()
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads

    self.w_init = variance_scaling(w_init_scale, 'average')

    self.query = nn.Linear(self.model_size, self.num_heads * self.key_size)
    self.w_init(self.query.weight)

    self.key = nn.Linear(self.model_size, self.num_heads * self.key_size)
    self.w_init(self.key.weight)

    self.value = nn.Linear(self.model_size, self.num_heads * self.value_size)
    self.w_init(self.value.weight)

    self.linear = nn.Linear(self.num_heads * self.key_size, self.model_size)
    self.w_init(self.linear.weight)

  def _linear_projection(
          self, y_linear: nn.Linear, x: torch.Tensor, head_size: int) -> torch.Tensor:
      # self, y_linear: nn.Linear, x: torch.Tensor, head_size: int) -> torch.Tensor:
    # y_linear = nn.Linear(self.model_size, self.num_heads * head_size)
    # self.w_init(y_linear.weight)
    y = y_linear(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, self.num_heads, head_size))
  
  def __call__(
      self,
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:

    # In shape hints below, we suppress the leading dims [...] for brevity.
    # Hence e.g. [A, B] should be read in every case as [..., A, B].
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection

    # Compute key/query/values (overload K/Q/V to denote the respective sizes).
    query_heads = projection(self.query, query, self.key_size)  # [T', H, Q=K]
    key_heads = projection(self.key, key, self.key_size)  # [T, H, K]
    value_heads = projection(self.value, value, self.value_size)  # [T, H, V]
  
    # Compute attention weights.
    attn_logits = torch.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / torch.sqrt(torch.tensor(self.key_size))
    attn_weights = torch.softmax(attn_logits, dim=-3)  # [H, T', T]

    # Weight the values by the attention and flatten the head vectors.
    attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    attn = torch.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]
    
    # Apply another projection to get the final embeddings.
    return self.linear(attn)  # [T', D']

