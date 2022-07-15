import math
from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, Bernoulli, TransformedDistribution
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn import functional as F
from .utils import Linear, get_parameters, get_named_parameters, ConvTranspose2DBlock, ResConv2DBlock
from .transformer_pytorch import TransformerEncoderLayer
import pdb

class TransformerModel(nn.Module):
  def __init__(self, cfg):
    super(TransformerModel, self).__init__()

    self.max_time = cfg.arch.transformer.max_time
    self.num_heads = cfg.arch.transformer.num_heads
    self.d_model = cfg.arch.transformer.d_model
    self.dim_feedforward = cfg.arch.transformer.dim_feedforward
    self.dropout = cfg.arch.transformer.dropout
    self.activation = cfg.arch.transformer.activation
    self.num_encoder_layers = cfg.arch.transformer.num_encoder_layers
    self.num_actions = cfg.env.action_size
    self.pos_enc = cfg.arch.transformer.pos_enc
    self.embedding_type = cfg.arch.transformer.embedding_type

    encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.num_heads,
                                               self.dim_feedforward, self.dropout,
                                               self.activation)
    self.transformer_encoder = nn.TransformerEncoder(
      encoder_layer, self.num_encoder_layers, norm=None)

    self.encode_s_type = cfg.arch.RSSM.encode_s_type

    if self.encode_s_type == 'dreamer':
      N = 1

    else:
      N = 32
      if self.embedding_type == 'linear':
        self.action_embedding = Linear(self.num_actions, self.d_model)
        self.input_embedding = Linear(cfg.arch.static_wm.vq_num_embeddings, self.d_model)

      if self.embedding_type == 'embedding':
        self.action_embedding = nn.Embedding(self.num_actions, self.d_model)
        self.input_embedding = nn.Embedding(cfg.arch.static_wm.vq_num_embeddings, self.d_model)

    if self.pos_enc == 'spatial':
      self.time_embedding = SinusoidalEncoding(self.d_model*N, self.max_time)
    if self.pos_enc == 'temporal':
      self.time_embedding = SinusoidalEncoding(self.d_model, self.max_time)

  def _generate_square_subsequent_mask(self, T, H, W, device):
    N = H * W
    mask = (torch.triu(torch.ones(T, T,
                                  device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-1e10')).masked_fill(
      mask == 1, float(0.0))

    mask = torch.repeat_interleave(mask, N, dim=0)
    mask = torch.repeat_interleave(mask, N, dim=1)

    return mask

  def forward(self, z, actions):
    B, T, D, H, W = z.shape

    attn_mask = self._generate_square_subsequent_mask(T, H, W, z.device)

    # (T, 1, d_model)
    if self.pos_enc == 'spatial':
      time_enc = self.time_embedding(torch.arange(T)).reshape(T, H, W, 1, self.d_model).reshape(T * H * W, 1,
                                                                                                self.d_model)
    else:
      time_enc = self.time_embedding(torch.arange(T * H * W))

    if actions is None:

      z = rearrange(z, 'b t d h w -> (t h w) b d')
      encoder_inp = z + time_enc

    else:
      z = rearrange(z, 'b t d h w -> (t h w) b d')
      actions = rearrange(actions, 'b t d -> t b d')

      if self.embedding_type == 'linear':
        encoder_inp = self.input_embedding(z) + time_enc
        action_emb = self.action_embedding(actions)
      else:
        # encoder_inp = self.input_embedding(torch.argmax(z, dim=-1)) + time_enc
        encoder_inp = z @ self.input_embedding.weight + time_enc
        # action_emb = self.action_embedding(actions.argmax(dim=-1).long())
        action_emb = actions @ self.action_embedding.weight
      action_emb = torch.repeat_interleave(action_emb, H*W, dim=0)
      encoder_inp += action_emb

    # T, B, d_model
    output = self.transformer_encoder(encoder_inp, mask=attn_mask)

    output = rearrange(output,
                       '(t h w) b d -> b t d h w',
                       h=H, w=W)
    return output

class SinusoidalEncoding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(SinusoidalEncoding, self).__init__()
    se = torch.zeros(max_len + 1, d_model)
    inp = torch.arange(0, max_len + 1, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, d_model, 2).float() *
      (-math.log(10000.0) / d_model))
    se[:, 0::2] = torch.sin(inp * div_term)
    se[:, 1::2] = torch.cos(inp * div_term)

    se = se.unsqueeze(1)
    self.register_buffer('se', se)

  def forward(self, x):
    return self.se[x, :]