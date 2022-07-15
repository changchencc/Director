"""
https://github.com/dhruvramani/Transformers-RL
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Linear

class PositionalEmbedding(torch.nn.Module):
  def __init__(self, dim):
    super(PositionalEmbedding, self).__init__()

    self.dim = dim
    inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
    self.register_buffer("inv_freq", inv_freq)

  def forward(self, positions):
    sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    d_model = cfg.arch.gtrxl.d_model
    d_inner = cfg.arch.gtrxl.d_ff_inner
    dropout = cfg.arch.gtrxl.dropout
    self.pre_lnorm = cfg.arch.gtrxl.pre_lnorm

    self.CoreNet = nn.Sequential(
      Linear(d_model, d_inner),
      nn.ReLU(inplace=True),
      Linear(d_inner, d_model),
      nn.Dropout(dropout)
    )

    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, inp):

    if self.pre_lnorm:
      ##### layer normalization + positionwise feed-forward
      output = self.CoreNet(self.layer_norm(inp))

    else:
      ##### positionwise feed-forward
      core_out = self.CoreNet(inp)

      ##### layer normalization
      output = self.layer_norm(core_out)

    return output


class GRUGatingMechanism(torch.nn.Module):
  def __init__(self, d_input, bg=0.1):
    super().__init__()
    self.Wr = Linear(d_input, d_input, bias=False)
    self.Ur = Linear(d_input, d_input, bias=False)
    self.Wz = Linear(d_input, d_input, bias=False)
    self.Uz = Linear(d_input, d_input)
    self.Wg = Linear(d_input, d_input, bias=False)
    self.Ug = Linear(d_input, d_input, bias=False)
    self.bg = bg

    self.sigmoid = torch.nn.Sigmoid()
    self.tanh = torch.nn.Tanh()

  def forward(self, x, y):
    r = self.sigmoid(self.Wr(y) + self.Ur(x))
    z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
    h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
    g = torch.mul(1 - z, x) + torch.mul(z, h)
    return g


class MultiheadAttentionXL(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    d_model = cfg.arch.gtrxl.d_model
    n_head = cfg.arch.gtrxl.n_head
    d_inner = cfg.arch.gtrxl.d_inner
    dropout = cfg.arch.gtrxl.dropout
    dropatt = cfg.arch.gtrxl.dropatt
    pre_lnorm = cfg.arch.gtrxl.pre_lnorm

    self.d_inner = d_inner
    self.n_head = n_head

    self.qkv_net = Linear(d_model, d_inner * n_head * 3, bias=False)
    self.out_net = Linear(d_inner * n_head, d_model, bias=False)
    self.r_net = Linear(d_model, d_inner * n_head, bias=False)

    self.drop = nn.Dropout(dropout)
    self.dropatt = nn.Dropout(dropatt)
    self.layer_norm = nn.LayerNorm(d_model)

    self.scale = 1 / (d_inner ** 0.5)

    self.pre_lnorm = pre_lnorm

  def _rel_shift(self, x):

    zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                           device=x.device, dtype=x.dtype)

    x_padded = torch.cat([zero_pad, x], dim=1)
    x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

    x = x_padded[1:].view_as(x)
    return x

  def forward(self, inpts, pos_emb, r_w_bias, r_r_bias, attn_mask=None, mems=None):
    """

    :param inpts: T, B, D
    :param pos_emb: T+Tm, B, D
    :param r_w_bias: n_head, d_head
    :param r_r_bias: n_head, d_head
    :param attn_mask: T, T+Tm
    :param mems: Tm, B, D
    :return:
    """

    q_len, r_len, bsz = inpts.size(0), pos_emb.size(0), inpts.size(1)

    if mems is not None:
      cat = torch.cat([mems, inpts], 0)
      if self.pre_lnorm:
        w_heads = self.qkv_net(self.layer_norm(cat))
      else:
        w_heads = self.qkv_net(cat)
      r_head_k = self.r_net(pos_emb)

      w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
      w_head_q = w_head_q[-q_len:]
    else:
      if self.pre_lnorm:
        w_heads = self.qkv_net(self.layer_norm(inpts))
      else:
        w_heads = self.qkv_net(inpts)
      r_head_k = self.r_net(pos_emb)

      w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

    k_len = w_head_k.size(0)
    w_head_q = w_head_q.view(q_len, bsz, self.n_head, self.d_inner)  # qlen x bsz x n_head x d_head
    w_head_k = w_head_k.view(k_len, bsz, self.n_head, self.d_inner)  # qlen x bsz x n_head x d_head
    w_head_v = w_head_v.view(k_len, bsz, self.n_head, self.d_inner)  # qlen x bsz x n_head x d_head

    r_head_k = r_head_k.view(r_len, self.n_head, self.d_inner)  # qlen x n_head x d_head

    #### compute attention score
    rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
    AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

    rr_head_q = w_head_q + r_r_bias
    BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
    BD = self._rel_shift(BD)

    # [qlen x klen x bsz x n_head]
    attn_score = AC + BD
    attn_score.mul_(self.scale)

    #### compute attention probability
    if attn_mask is not None and attn_mask.any().item():
      if attn_mask.dim() == 2:
        attn_score = attn_score.float().masked_fill(
          attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
      elif attn_mask.dim() == 3:
        attn_score = attn_score.float().masked_fill(
          attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

    # [qlen x klen x bsz x n_head]
    attn_prob = F.softmax(attn_score, dim=1)
    attn_prob = self.dropatt(attn_prob)

    #### compute attention vector
    attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

    # [qlen x bsz x n_head x d_head]
    attn_vec = attn_vec.contiguous().view(
      attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_inner)

    ##### linear projection
    attn_out = self.out_net(attn_vec)
    attn_out = self.drop(attn_out)

    if self.pre_lnorm:
      ##### residual connection
      output = attn_out
    else:
      ##### residual connection + layer normalization
      output = self.layer_norm(attn_out)

    return output

class GTrXLEncoderLayer(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    d_model = cfg.arch.gtrxl.d_model
    self.gating = cfg.arch.gtrxl.gating

    self.gate1 = GRUGatingMechanism(d_model)
    self.gate2 = GRUGatingMechanism(d_model)
    self.mah = MultiheadAttentionXL(cfg)
    self.pos_ff = PositionwiseFF(cfg)

  def forward(self, inpts, pos_emb, r_w_bias, r_r_bias, attn_mask=None, mems=None):

    src2 = self.mah(inpts, pos_emb, r_w_bias, r_r_bias, attn_mask=attn_mask, mems=mems)
    src = self.gate1(inpts, src2) if self.gating else inpts + src2

    src2 = self.pos_ff(src)
    src = self.gate2(src, src2) if self.gating else src + src2

    return src

class GTrXL(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    d_model = cfg.arch.gtrxl.d_model
    n_head = cfg.arch.gtrxl.n_head
    d_inner = cfg.arch.gtrxl.d_inner
    n_layers = cfg.arch.gtrxl.n_layers
    dropout = cfg.arch.gtrxl.dropout
    self.deter_type = cfg.arch.gtrxl.deter_type
    self.mem_len = cfg.arch.gtrxl.mem_len
    self.d_model = d_model
    self.n_layers = n_layers

    self.pos_embs = PositionalEmbedding(d_model)
    self.drop = torch.nn.Dropout(dropout)

    self.layers = torch.nn.ModuleList(
      [GTrXLEncoderLayer(cfg) for _ in range(n_layers)]
    )
    self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_inner))
    self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_inner))
    nn.init.normal_(self.r_w_bias, 0.0, 0.02)
    nn.init.normal_(self.r_r_bias, 0.0, 0.02)

  def init_memory(self, bs, device=torch.device("cpu")):
    return [
      # torch.empty(0, dtype=torch.float).to(device)
      torch.zeros(self.mem_len, bs, self.d_model, dtype=torch.float).to(device)
      for _ in range(self.n_layers + 1)
    ]

  def update_memory(self, previous_memory, hidden_states):
    """
    + Arguments
        - previous_memory: List[torch.FloatTensor],
        - hidden_states: List[torch.FloatTensor]
    """
    assert len(hidden_states) == len(previous_memory)
    mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
    # mem_len, seq_len = 3, hidden_states[0].size(0)
    # print(mem_len, seq_len)

    with torch.no_grad():
      new_memory = []
      end_idx = mem_len + seq_len
      beg_idx = max(0, end_idx - mem_len)
      for m, h in zip(previous_memory, hidden_states):
        cat = torch.cat([m, h], dim=0)
        new_memory.append(cat[beg_idx:end_idx].detach())
    return new_memory

  def forward(self, inputs, memory=None):
    """
    + Arguments
        - inputs - torch.FloatTensor = [T x B x d_inner]
        - memory - Optional, list[torch.FloatTensor] = [[T x B x d_inner] x 5]
    """
    q_len, bsz = inputs.shape[:2]

    if memory is None:
      memory = self.init_memory(bsz, inputs.device)
    assert len(memory) == len(self.layers) + 1

    m_len = memory[0].size(0)
    k_len = m_len + q_len

    # dec_attn_mask = [curr x curr + prev x 1] = [20 x 40 x 1]
    dec_attn_mask = torch.triu(
        torch.ones((q_len, k_len)),
        diagonal=1 + m_len).bool()[..., None].to(inputs.device)

    pos_ips = torch.arange(k_len-1, -1, -1.0, dtype=torch.float).to(inputs.device)
    # pos_embs = [curr + prev x 1 x d_input] = [40 x 1 x 8]
    pos_embs = self.drop(self.pos_embs(pos_ips))
    if self.d_model % 2 != 0:
      pos_embs = pos_embs[:, :, :-1]

    hidden_states = [inputs]
    layer_out = inputs
    for mem, layer in zip(memory, self.layers):
      # layer_out = [curr x B x d_inner] = [20 x 5 x 8]
      layer_out = layer(
        layer_out,
        pos_embs,
        self.r_w_bias,
        self.r_r_bias,
        attn_mask=dec_attn_mask,
        mems=mem,
      )
      hidden_states.append(layer_out)

    #todo: check this dropout
    # layer_out = self.drop(layer_out)

    # Memory is treated as a const., don't propagate through it
    # new_memory = [[T x B x d_inner] x 4]
    memory = self.update_memory(memory, hidden_states)
    if self.deter_type == 'concat_o':
      layer_out = torch.cat(hidden_states, dim=-1)
    return {"logits": layer_out, "memory": memory}