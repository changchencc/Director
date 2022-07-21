import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli
import pdb
import numpy as np


class SubConv2DBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        k,
        s,
        p,
        num_groups,
        bias=True,
        non_linearity=True,
        weight_init="xavier",
        act="relu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=k, stride=1, padding=p, bias=bias
        )
        self.pix_shuffle = nn.PixelShuffle(s)

        if weight_init == "xavier":
            nn.init.xavier_uniform_(self.conv.weight)
        else:
            nn.init.kaiming_uniform_(self.conv.weight)

        if bias:
            nn.init.zeros_(self.conv.bias)

        if num_groups > 0:
            self.group_norm = nn.GroupNorm(
                num_groups=num_groups, num_channels=c_out // s // s
            )

        if non_linearity:
            if act == "relu":
                self.non_linear = nn.ReLU()
            else:
                self.non_linear = nn.CELU()

        self.non_linearity = non_linearity
        self.num_groups = num_groups

    def forward(self, inputs):

        o = self.conv(inputs)
        o = self.pix_shuffle(o)

        if self.num_groups > 0:
            o = self.group_norm(o)

        if self.non_linearity:
            o = self.non_linear(o)

        return o


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        k,
        s,
        p,
        num_groups=0,
        bias=True,
        non_linearity=True,
        weight_init="xavier",
        act="relu",
    ):
        super().__init__()

        self.net = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)

        if weight_init == "xavier":
            nn.init.xavier_uniform_(self.net.weight)
        else:
            nn.init.kaiming_uniform_(self.net.weight)

        if bias:
            nn.init.zeros_(self.net.bias)

        if num_groups > 0:
            self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)

        if non_linearity:
            if act == "relu":
                self.non_linear = nn.ReLU()
            elif act == "elu":
                self.non_linear = nn.ELU()
            else:
                self.non_linear = nn.CELU()

        self.non_linearity = non_linearity
        self.num_groups = num_groups

    def forward(self, inputs):

        o = self.net(inputs)

        if self.num_groups > 0:
            o = self.group_norm(o)

        if self.non_linearity:
            o = self.non_linear(o)

        return o


class ConvTranspose2DBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        k,
        s,
        p,
        num_groups=0,
        bias=True,
        non_linearity=True,
        weight_init="xavier",
        act="relu",
    ):
        super().__init__()

        self.net = nn.ConvTranspose2d(
            c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias
        )

        if weight_init == "xavier":
            nn.init.xavier_uniform_(self.net.weight)
        else:
            nn.init.kaiming_uniform_(self.net.weight)

        if bias:
            nn.init.zeros_(self.net.bias)

        if num_groups > 0:
            self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)

        if non_linearity:
            if act == "relu":
                self.non_linear = nn.ReLU()
            elif act == "elu":
                self.non_linear = nn.ELU()
            else:
                self.non_linear = nn.CELU()

        self.non_linearity = non_linearity
        self.num_groups = num_groups

    def forward(self, inputs):

        o = self.net(inputs)

        if self.num_groups > 0:
            o = self.group_norm(o)

        if self.non_linearity:
            o = self.non_linear(o)

        return o


# class ResConv2DBlock(nn.Module):
#   def __init__(self, c_in, c_out, num_groups,
#                    weight_init='xavier', act='relu'):
#     super().__init__()
#
#     self.residuel = nn.Sequential(
#       Conv2DBlock(c_in, c_out//2, 3, 1, 1, num_groups,
#                   weight_init=weight_init, act=act),
#       Conv2DBlock(c_out//2, c_out, 3, 1, 1, 0,
#                   non_linearity=False, weight_init=weight_init)
#     )
#
#     self.skip = Conv2DBlock(c_in, c_out, 3, 1, 1, num_groups,
#                             non_linearity=False, weight_init=weight_init)
#
#     if num_groups > 0:
#       self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)
#     if act == 'relu':
#       self.non_linear = nn.ReLU()
#     else:
#       self.non_linear = nn.CELU()
#
#   def forward(self, inputs):
#     return self.non_linear(self.group_norm(self.residuel(inputs) + self.skip(inputs)))


class ResConv2DBlock(nn.Module):
    def __init__(self, c_in, c_out, num_groups, weight_init="xavier", act="relu"):
        super().__init__()

        self.residuel = nn.Sequential(
            nn.ReLU(),
            Conv2DBlock(
                c_in, c_out // 2, 3, 1, 1, num_groups, weight_init=weight_init, act=act
            ),
            Conv2DBlock(
                c_out // 2,
                c_out,
                1,
                1,
                0,
                0,
                non_linearity=False,
                weight_init=weight_init,
            ),
        )

        self.skip = Conv2DBlock(
            c_in,
            c_out,
            3,
            1,
            1,
            num_groups,
            non_linearity=False,
            weight_init=weight_init,
        )

        if num_groups > 0:
            self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)
        if act == "relu":
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = nn.CELU()

    def forward(self, inputs):
        return self.non_linear(
            self.group_norm(self.residuel(inputs) + self.skip(inputs))
        )


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, weight_init="xavier"):
        super().__init__()

        self.net = nn.Linear(dim_in, dim_out, bias=bias)

        if weight_init == "xavier":
            nn.init.xavier_uniform_(self.net.weight)
        else:
            nn.init.kaiming_uniform_(self.net.weight)

        if bias:
            nn.init.zeros_(self.net.bias)

    def forward(self, inputs):
        return self.net(inputs)


class MLP(nn.Module):
    def __init__(self, dims, act, weight_init, output_act=None, norm=False):
        super().__init__()

        dims_in = dims[:-2]
        dims_out = dims[1:-1]

        layers = []
        for d_in, d_out in zip(dims_in, dims_out):
            layers.append(Linear(d_in, d_out, weight_init=weight_init, bias=True))
            if norm:
                layers.append(nn.LayerNorm(d_out))
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.CELU())

        layers.append(Linear(d_out, dims[-1], weight_init=weight_init, bias=True))
        if output_act:
            if norm:
                layers.append(nn.LayerNorm(dims[-1]))
            if act == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.CELU())

        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        return self.enc(x)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)

        nn.init.xavier_uniform_(self.gru_cell.weight_ih)
        nn.init.orthogonal_(self.gru_cell.weight_hh)

        if bias:
            nn.init.zeros_(self.gru_cell.bias_ih)
            nn.init.zeros_(self.gru_cell.bias_hh)

        self.h0 = nn.Parameter(torch.randn(1, hidden_size))
        torch.nn.init.kaiming_uniform_(self.h0)

    def init_state(self, batch_size):

        return self.h0.expand(batch_size, -1)

    def forward(self, x, h):
        """
    GRU for slot attention.
    inputs:
      x: (bs, c)
      h: (bs, h)
    """

        output_shape = h.shape
        x = x.reshape(-1, x.shape[-1])
        h = h.reshape(-1, h.shape[-1])

        h = self.gru_cell(x, h)
        h = h.reshape(output_shape)

        return h


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        nn.init.zeros_(self.rnn_cell.bias_ih)
        nn.init.zeros_(self.rnn_cell.bias_hh)

    def forward(self, x, h):
        """
    GRU for slot attention.
    inputs:
      x: (bs, N, c)
      h: (bs, N, h)
    """
        assert x.dim() == 3, "dim of input for GRUCell should be 3"
        assert h.dim() == 3, "dim of input for GRUCell should be 3"

        output_shape = h.shape
        x = x.reshape(-1, x.shape[-1])
        h = h.reshape(-1, h.shape[-1])

        h = self.rnn_cell(x, h)
        h = h.reshape(output_shape)

        return h


class ConvLayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, init_state=False):
        super().__init__()

        self.conv_i2h = Conv2DBlock(
            input_size, 2 * hidden_size, 3, 1, 1, 0, bias=False, non_linearity=False
        )  # we have layernorm, bias is redundant
        self.conv_h2h = Conv2DBlock(
            hidden_size, 2 * hidden_size, 3, 1, 1, 0, bias=False, non_linearity=False
        )
        self.conv_i2c = Conv2DBlock(
            input_size, hidden_size, 3, 1, 1, 0, bias=False, non_linearity=False
        )
        self.conv_h2c = Conv2DBlock(
            hidden_size, hidden_size, 3, 1, 1, 0, bias=False, non_linearity=False
        )

        self.norm_hh = nn.GroupNorm(1, 2 * hidden_size)
        self.norm_ih = nn.GroupNorm(1, 2 * hidden_size)
        self.norm_c = nn.GroupNorm(1, hidden_size)
        self.norm_u = nn.GroupNorm(1, hidden_size)

        nn.init.xavier_uniform_(self.conv_i2h.net.weight)
        nn.init.xavier_uniform_(self.conv_i2c.net.weight)
        nn.init.orthogonal_(self.conv_h2h.net.weight)
        nn.init.orthogonal_(self.conv_h2c.net.weight)

        if init_state:
            self.h0 = nn.Parameter(torch.randn(1, hidden_size, 4, 4))
            torch.nn.init.kaiming_uniform_(self.h0)

    def init_state(self, shape):

        bs = shape[0]

        return self.h0.expand(bs, -1, -1, -1)

    def forward(self, x, h):
        """
    GRU for slot attention.
    inputs:
      x: (bs, C, H, W)
      h: (bs, C, H, W)
    """

        if h is None:
            h = self.init_state(x.shape)

        assert x.dim() == 4, "dim of input for GRUCell should be 3"
        assert h.dim() == 4, "dim of input for GRUCell should be 3"

        output_shape = h.shape

        i2h = self.conv_i2h(x)
        h2h = self.conv_h2h(h)

        logits_zr = self.norm_ih(i2h) + self.norm_hh(h2h)
        z, r = logits_zr.chunk(2, dim=1)

        i2c = self.conv_i2c(x)
        h2c = self.conv_h2c(h)

        c = (self.norm_c(i2c) + r.sigmoid() * self.norm_u(h2c)).tanh()

        h = (1.0 - z.sigmoid()) * h + z.sigmoid() * c

        h = h.reshape(output_shape)

        return h


class LayerNormGRUCellV2(nn.Module):
    """
  This is used in dreamerV2.
  """

    def __init__(self, input_size, hidden_size, bias=False):
        super().__init__()
        input_size = input_size + hidden_size

        self.fc = Linear(
            input_size, 3 * hidden_size, bias=bias
        )  # we have layernorm, bias is redundant
        # self.fc_h2h = Linear(hidden_size, 2*hidden_size, bias=False)
        # self.fc_i2c = Linear(input_size, hidden_size, bias=False)
        # self.fc_h2c = Linear(hidden_size, hidden_size, bias=False)

        self.layer_norm = nn.LayerNorm(3 * hidden_size)
        # self.layer_norm_ih = nn.LayerNorm(2*hidden_size)
        # self.layer_norm_c = nn.LayerNorm(hidden_size)
        # self.layer_norm_u = nn.LayerNorm(hidden_size)

        nn.init.xavier_uniform_(self.fc.net.weight)
        # nn.init.xavier_uniform_(self.fc_i2c.net.weight)
        # nn.init.orthogonal_(self.fc_h2h.net.weight)
        # nn.init.orthogonal_(self.fc_h2c.net.weight)

        self.h0 = nn.Parameter(torch.randn(1, hidden_size))
        torch.nn.init.kaiming_uniform_(self.h0)

    def init_state(self, batch_size):

        return self.h0.expand(batch_size, -1)

    def forward(self, x, h):
        """
    GRU for slot attention.
    inputs:
      x: (bs, N, c), or (bs, C)
      h: (bs, N, h), or (bs, C)
    """

        if h is None:
            h = self.init_state(x.shape)

        assert x.dim() == 3 or x.dim() == 2, "dim of input for GRUCell should be 3 or 2"
        assert h.dim() == 3 or x.dim() == 2, "dim of input for GRUCell should be 3 or 2"

        output_shape = h.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
            h = h.reshape(-1, h.shape[-1])

        logits = self.fc(torch.cat([x, h], dim=-1))
        logits = self.layer_norm(logits)

        r, c, u = logits.chunk(3, dim=-1)

        r = r.sigmoid()
        c = (r * c).tanh()
        u = (u - 1.0).sigmoid()

        h = u * c + (1.0 - u) * h
        h = h.reshape(output_shape)

        return h


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.gamma = nn.Parameter(torch.Tensor(*normalized_shape))
        self.beta = nn.Parameter(torch.Tensor(*normalized_shape))
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)
        self.eps = eps

    def forward(self, inpts):

        try:
            mean = inpts.mean(dim=-1, keepdim=True)
            std = inpts.std(dim=-1, keepdim=True)
            normed = (inpts - mean) / (std + self.eps).sqrt()
            o = normed * self.gamma + self.beta
        except:
            pdb.set_trace()

        return o


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc_i2h = Linear(
            input_size, 2 * hidden_size, bias=False
        )  # we have layernorm, bias is redundant
        self.fc_h2h = Linear(hidden_size, 2 * hidden_size, bias=False)
        self.fc_i2c = Linear(input_size, hidden_size, bias=False)
        self.fc_h2c = Linear(hidden_size, hidden_size, bias=False)

        self.layer_norm_hh = nn.LayerNorm(2 * hidden_size)
        self.layer_norm_ih = nn.LayerNorm(2 * hidden_size)
        self.layer_norm_c = nn.LayerNorm(hidden_size)
        self.layer_norm_u = nn.LayerNorm(hidden_size)

        nn.init.xavier_uniform_(self.fc_i2h.net.weight)
        nn.init.xavier_uniform_(self.fc_i2c.net.weight)
        nn.init.orthogonal_(self.fc_h2h.net.weight)
        nn.init.orthogonal_(self.fc_h2c.net.weight)

        self.h0 = nn.Parameter(torch.randn(1, hidden_size))
        torch.nn.init.kaiming_uniform_(self.h0)

    def init_state(self, batch_size):

        return self.h0.expand(batch_size, -1)

    def forward(self, x, h):
        """
    GRU for slot attention.
    inputs:
      x: (bs, N, c), or (bs, C)
      h: (bs, N, h), or (bs, C)
    """

        if h is None:
            h = self.init_state(x.shape)

        assert x.dim() == 3 or x.dim() == 2, "dim of input for GRUCell should be 3 or 2"
        assert h.dim() == 3 or x.dim() == 2, "dim of input for GRUCell should be 3 or 2"

        output_shape = h.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
            h = h.reshape(-1, h.shape[-1])

        i2h = self.fc_i2h(x)
        h2h = self.fc_h2h(h)

        logits_zr = self.layer_norm_ih(i2h) + self.layer_norm_hh(h2h)
        z, r = logits_zr.chunk(2, dim=-1)

        i2c = self.fc_i2c(x)
        h2c = self.fc_h2c(h)

        c = (self.layer_norm_c(i2c) + r.sigmoid() * self.layer_norm_u(h2c)).tanh()

        h_new = (1.0 - z.sigmoid()) * h + z.sigmoid() * c

        h_new = h_new.reshape(output_shape)

        return h_new


class ConvLSTMCell(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, spatial_size, k=3, p=1):
        super().__init__()

        weight_init = cfg.arch.weight_init

        self.conv_ = Conv2DBlock(
            input_size + hidden_size,
            4 * hidden_size,
            k,
            1,
            p,
            0,
            bias=False,
            non_linearity=False,
            weight_init=weight_init,
        )

        self.bias = nn.Parameter(
            torch.zeros(1, 4 * hidden_size, 1, 1), requires_grad=True
        )

        self.h0, self.c0 = self.init_state(hidden_size, spatial_size, weight_init)

    def init_state(self, hidden_size, spatial_size, weight_init):

        h0 = torch.randn(1, hidden_size, spatial_size, spatial_size)
        c0 = torch.randn(1, hidden_size, spatial_size, spatial_size)

        nn.init.zeros_(h0)
        nn.init.zeros_(c0)

        return (
            nn.Parameter(h0, requires_grad=True),
            nn.Parameter(c0, requires_grad=True),
        )

    def forward(self, x, state):
        """
    LSTM for global attention decoder.
    """

        if state is None:
            bs = x.shape[0]

            h, c = self.h0, self.c0
            h = h.expand(bs, -1, -1, -1)
            c = c.expand(bs, -1, -1, -1)

        else:
            h, c = state

        logits = self.conv_(torch.cat([x, h], dim=1)) + self.bias

        f, i, o, g = logits.chunk(4, dim=1)

        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()

        return h, c


class ConvLayerNormLSTMCell(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, spatial_size, k=3, p=1):
        super().__init__()

        weight_init = cfg.arch.weight_init

        self.conv_i2h = Conv2DBlock(
            input_size,
            4 * hidden_size,
            k,
            1,
            p,
            0,
            bias=False,
            non_linearity=False,
            weight_init=weight_init,
        )
        self.conv_h2h = Conv2DBlock(
            hidden_size,
            4 * hidden_size,
            k,
            1,
            p,
            0,
            bias=False,
            non_linearity=False,
            weight_init=weight_init,
        )
        # self.conv_i2h = nn.Conv2d(input_size, 4*hidden_size, k, 1, p, bias=False)
        # self.conv_h2h = nn.Conv2d(hidden_size, 4*hidden_size, k, 1, p, bias=False)

        self.norm_h = nn.GroupNorm(1, 4 * hidden_size)
        self.norm_i = nn.GroupNorm(1, 4 * hidden_size)
        self.norm_c = nn.GroupNorm(1, hidden_size)

        self.bias = nn.Parameter(
            torch.zeros(1, 4 * hidden_size, 1, 1), requires_grad=True
        )

        self.h0, self.c0 = self.init_state(hidden_size, spatial_size, weight_init)

    def init_state(self, hidden_size, spatial_size, weight_init):

        h0 = torch.randn(1, hidden_size, spatial_size, spatial_size)
        c0 = torch.randn(1, hidden_size, spatial_size, spatial_size)

        nn.init.zeros_(h0)
        nn.init.zeros_(c0)

        # if weight_init == 'xavier':
        #   nn.init.xavier_uniform_(h0)
        #   nn.init.xavier_uniform_(c0)
        #
        # else:
        #   nn.init.kaiming_uniform_(h0)
        #   nn.init.kaiming_uniform_(c0)

        return (
            nn.Parameter(h0, requires_grad=True),
            nn.Parameter(c0, requires_grad=True),
        )

    def forward(self, x, state):
        """
    LSTM for global attention decoder.
    """

        if state is None:
            bs = x.shape[0]

            h, c = self.h0, self.c0
            h = h.expand(bs, -1, -1, -1)
            c = c.expand(bs, -1, -1, -1)

        else:
            h, c = state

        i2h = self.conv_i2h(x)
        h2h = self.conv_h2h(h)

        logits = self.norm_i(i2h) + self.norm_h(h2h) + self.bias
        # logits = i2h + h2h + self.bias

        f, i, o, g = logits.chunk(4, dim=1)

        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * self.norm_c(c).tanh()

        return h, c


class LayerNormLSTMCell(nn.Module):
    def __init__(self, cfg, input_size, hidden_size):
        super().__init__()

        self.fc_i2h = Linear(
            input_size, 4 * hidden_size, weight_init=cfg.arch.weight_init, bias=False
        )
        self.fc_h2h = Linear(
            hidden_size, 4 * hidden_size, weight_init=cfg.arch.weight_init, bias=False
        )

        self.layer_norm_h = nn.LayerNorm(4 * hidden_size)
        self.layer_norm_i = nn.LayerNorm(4 * hidden_size)
        self.layer_norm_c = nn.LayerNorm(hidden_size)

        self.bias = nn.Parameter(torch.zeros(4 * hidden_size), requires_grad=True)

        self.h0, self.c0 = self.init_state(hidden_size)

    def init_state(self, hidden_size):

        h0 = torch.randn(1, hidden_size)
        c0 = torch.randn(1, hidden_size)

        nn.init.zeros_(h0)
        nn.init.zeros_(c0)

        return (
            nn.Parameter(h0, requires_grad=True),
            nn.Parameter(c0, requires_grad=True),
        )

    def forward(self, x, state):
        """
    LSTM for global attention decoder.
    """

        h, c = state

        i2h = self.fc_i2h(x)
        h2h = self.fc_h2h(h)

        logits = self.layer_norm_i(i2h) + self.layer_norm_h(h2h) + self.bias

        f, i, o, g = logits.chunk(4, dim=-1)

        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * self.layer_norm_c(c).tanh()

        return h, c


class GroupLSTMCell(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, num_units):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_units = num_units

        self.i2h = Linear(
            num_units * input_size,
            4 * num_units * hidden_size,
            weight_init=cfg.arch.weight_init,
            bias=False,
        )
        self.h2h = Linear(
            num_units * hidden_size,
            4 * num_units * hidden_size,
            weight_init=cfg.arch.weight_init,
            bias=False,
        )

        self.bias = nn.Parameter(
            torch.zeros(1, num_units, 4 * hidden_size), requires_grad=True
        )
        self.h0, self.c0 = self.init_states()

    def init_states(self):
        h = torch.randn(1, self.num_units, self.hidden_size)
        c = torch.randn(1, self.num_units, self.hidden_size)
        nn.init.zeros_(h)
        nn.init.zeros_(c)

        return nn.Parameter(h, requires_grad=True), nn.Parameter(c, requires_grad=True)

    def forward(self, x, h, c):
        """
    x: bs, num_units, C
    h, c: bs, num_units, H
    """

        i2h = self.i2h(x.reshape(-1, self.num_units * self.input_size))
        h2h = self.h2h(h.reshape(-1, self.num_units * self.hidden_size))

        i2h = i2h.reshape(-1, self.num_units, 4 * self.hidden_size)
        h2h = h2h.reshape(-1, self.num_units, 4 * self.hidden_size)

        logits = i2h + h2h + self.bias

        f, i, o, g = logits.chunk(4, dim=-1)

        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()

        return h, c


def get_position_grid4d(h, w, bs, device):

    x_0 = (
        torch.linspace(0, w, steps=w, dtype=torch.float, device=device) / w
    )  # distance to the left border
    x_1 = torch.flip(x_0, dims=(0,))  # distance to the right border
    y_0 = (
        torch.linspace(0, h, steps=h, dtype=torch.float, device=device) / h
    )  # distance to the upper border
    y_1 = torch.flip(y_0, dims=(0,))  # distance to the bottom border

    x_0 = x_0.reshape(1, 1, w).expand(bs, h, -1)
    x_1 = x_1.reshape(1, 1, w).expand(bs, h, -1)
    y_0 = y_0.reshape(1, h, 1).expand(bs, -1, w)
    y_1 = y_1.reshape(1, h, 1).expand(bs, -1, w)

    pos_grid = torch.stack([x_0, x_1, y_0, y_1], dim=3)

    return pos_grid


def get_position_grid2d(h, w, bs, device):

    x_0 = (
        2.0 * torch.linspace(0, w, steps=w, dtype=torch.float, device=device) / w - 1.0
    )  # distance to the left border
    y_0 = (
        2.0 * torch.linspace(0, h, steps=h, dtype=torch.float, device=device) / h - 1.0
    )  # distance to the upper border
    # x_0 = torch.linspace(0, w, steps=w, dtype=torch.float, device=device) / w # distance to the left border
    # y_0 = torch.linspace(0, h, steps=h, dtype=torch.float, device=device) / h # distance to the upper border

    x_0 = x_0.reshape(1, 1, w).expand(bs, h, -1)
    y_0 = y_0.reshape(1, h, 1).expand(bs, -1, w)

    pos_grid = torch.stack([x_0, y_0], dim=1)

    return pos_grid


def weighted_mean(attn, value):

    weighted_attn = attn / attn.sum(dim=1, keepdim=True).detach()

    return weighted_attn, torch.bmm(weighted_attn.permute(0, 2, 1), value)


def spatial_transformer(pos, scale, imgs, img_h, img_w, mode="bilinear", inverse=False):
    """
  Applying spatial transformer on images given pos ans scale.
  :param pos: tensor of size (N, 2)
  :param scale:  tensor of size (N, 2)
  :param img: tensor of size (N, 3, IMG_H, IMG_W)
  :return:
  """

    assert (
        pos.shape[0] == scale.shape[0]
    ), "pos and scale should have the same number on dimension 0"
    assert (
        pos.shape[0] == imgs.shape[0]
    ), "pos and images should have the same number on dimension 0"

    N = pos.shape[0]
    img_c = imgs.shape[1]

    if inverse:

        pos = -pos / scale
        scale = 1.0 / scale

    theta = pos.new_zeros(N, 2, 3)
    theta[:, :, -1] = pos.reshape(N, 2)
    theta[:, 0, 0] = scale[:, 0].reshape(N)
    theta[:, 1, 1] = scale[:, 1].reshape(N)

    grid = F.affine_grid(
        theta.view(N, 2, 3), [N, img_c, img_h, img_w], align_corners=False
    )
    patches = F.grid_sample(imgs, grid, mode=mode, align_corners=False)
    return patches


def get_offset(grid_h, grid_w):
    h = torch.arange(grid_h)
    w = torch.arange(grid_w)
    return torch.meshgrid(h, w)


def st_pos(xy):
    H, W = xy.shape[2:]
    x_offsets, y_offsets = get_offset(H, W)
    x = (
        (0.4 * (xy[:, :1]).tanh() - 0.45).exp()
        - 1.45
        + 0.5 * y_offsets.to(xy.device).view(1, 1, H, W)
    )
    y = (
        (0.4 * (xy[:, 1:]).tanh() - 0.45).exp()
        - 1.45
        + 0.5 * x_offsets.to(xy.device).view(1, 1, H, W)
    )
    return torch.cat([x, y], dim=1)


def st_scale(wh, scale_factor):
    w = (scale_factor - wh[:, :1].tanh()).exp()
    h = (scale_factor - wh[:, 1:].tanh()).exp()
    return torch.cat([w, h], dim=1)


def kl_divergence_relax_bern(q, p, eps=1e-15):
    q_probs = q.probs
    p_probs = p.probs
    kl_loss = q_probs * ((q_probs + eps).log() - (p_probs + eps).log()) + (
        1.0 - q_probs
    ) * ((1.0 - q_probs + eps).log() - (1.0 - p_probs + eps).log())
    return kl_loss


class MyRelaxedBernoulli(RelaxedBernoulli):
    def __init__(self, temp, logits=None, probs=None):
        super(MyRelaxedBernoulli, self).__init__(temp, probs=probs, logits=logits)
        """
    re-write the rsample() api.
    """
        if logits is None:
            self.device = probs.device
        if probs is None:
            self.device = logits.device

    def rsample(self, shape=None, eps=1e-15):
        if shape is None:
            shape = self.probs.shape
        uniforms = torch.rand(shape, dtype=self.logits.dtype, device=self.device)
        uniforms = torch.clamp(uniforms, eps, 1.0 - eps)
        samples = (
            (uniforms).log() - (-uniforms).log1p() + self.logits
        ) / self.temperature
        if torch.isnan(samples).any():
            pdb.set_trace()
        return samples

    def log_prob(self, values):
        diff = self.logits - values.mul(self.temperature)

        l_p = self.temperature.log() + diff - 2 * diff.exp().log1p()
        if torch.isinf(l_p).any():
            pdb.set_trace()
        return l_p


def linear_annealing(step, start_step, end_step, start_value, end_value):
    """
  Linear annealing

  :param x: original value. Only for getting device
  :param step: current global step
  :param start_step: when to start changing value
  :param end_step: when to stop changing value
  :param start_value: initial value
  :param end_value: final value
  :return:
  """
    if step <= start_step:
        x = start_value
    elif start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = start_value + slope * (step - start_step)
    else:
        x = end_value

    return x


def up_and_down_linear_schedule(
    step, start_step, mid_step, end_step, start_value, mid_value, end_value
):
    if start_step < step <= mid_step:
        slope = (mid_value - start_value) / (mid_step - start_step)
        x = start_value + slope * (step - start_step)
    elif mid_step < step < end_step:
        slope = (end_value - mid_value) / (end_step - mid_step)
        x = mid_value + slope * (step - mid_step)
    elif step >= end_step:
        x = end_value
    else:
        x = start_value

    return x


class GatedCNN(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, spatial_size):
        super().__init__()

        if spatial_size == 1:
            k = 1
            p = 0
        else:
            k = 3
            p = 1

        weight_init = cfg.arch.weight_init
        self.conv_h = Conv2DBlock(
            input_size + hidden_size,
            hidden_size * 2,
            k,
            1,
            p,
            0,
            non_linearity=False,
            weight_init=weight_init,
        )

    def init_state(self, hidden_size, spatial_size):

        h0 = torch.randn(1, hidden_size, spatial_size, spatial_size)

        nn.init.zeros_(h0)

        return nn.Parameter(h0, requires_grad=True)

    def forward(self, x, h):

        h = self.conv_h(torch.cat([x, h], dim=1))

        h1, h2 = torch.chunk(h, 2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)

        return h


def get_parameters(modules):
    """
  https://github.com/juliusfrost/dreamer-pytorch/blob/47bd509ab5cffa95ec613fd788d7ae1fe664ecd5/dreamer/utils/module.py
  Given a list of torch modules, returns a list of their parameters.
  :param modules: iterable of modules
  :returns: a list of parameters
  """
    model_parameters = []
    for k, module in modules.items():
        model_parameters += list(module.parameters())

    return model_parameters


class FreezeParameters:
    def __init__(self, modules_list):
        """
    https://github.com/juliusfrost/dreamer-pytorch/blob/47bd509ab5cffa95ec613fd788d7ae1fe664ecd5/dreamer/utils/module.py
    Context manager to locally freeze gradients.
    In some cases with can speed up computation because gradients aren't calculated for these listed modules.
    example:
    ```
    with FreezeParameters([module]):
        output_tensor = module(input_tensor)
    ```
    :param modules: iterable of modules. used to call .parameters() to freeze gradients.
    """
        self.modules = modules_list
        model_params_list = [get_parameters_dict(m) for m in self.modules]
        self.param_states = []
        for i, m in enumerate(model_params_list):
            param_states = dict()

            for k, params in m.items():
                param_states[k] = [p.requires_grad for p in params]

            self.param_states.append(param_states)
        # self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):

        for i, m in enumerate(self.modules):
            for k, m in m.items():
                for p in m.parameters():
                    p.requires_grad = False

            # for param in get_parameters(self.modules_list):
            #   param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):

        for i, m in enumerate(self.modules):
            for k, mm in m.items():
                for j, p in enumerate(mm.parameters()):
                    p.requires_grad = self.param_states[i][k][j]

        # for i, param in enumerate(get_parameters(self.modules)):
        #   param.requires_grad = self.param_states[i]


def get_parameters_dict(modules):
    """
  Given a dict of torch modules, returns a dict of their parameters.
  :param modules: iterable of modules
  :returns: a list of parameters
  """

    model_parameters = dict()
    assert isinstance(modules, dict), "only support dictionary in get_params."
    for k, module in modules.items():
        model_parameters[k] = list(module.parameters())

    return model_parameters


def get_named_parameters(modules, name):

    assert isinstance(modules, dict), "only support dictionary in get_named_params."
    named_parameters = dict()
    for k, module in modules.items():
        for n, p in module.named_parameters():
            named_parameters[f"{name}-{k}-{n}"] = p

    return named_parameters


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


if __name__ == "__main__":
    dims = [64, 64, 4]
    net = MLP(dims, nn.ReLU)
    pdb.set_trace()

