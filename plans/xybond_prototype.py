"""Parity-tracking bond net: two streams, EVEN and ODD, on the bond grid.

PROTOTYPE, not part of the `eesi` package. Measured against XYChainGNN in
plans/XY_BONDNET_REVIEW.md; land it as eesi/models/xybond.py only after the
acceptance criteria in Sec. 7 of that report are agreed.

Drop-in for XYChainGNN: forward(t [B] or scalar, x [B, N] or [B, N, 1]) -> same
shape as x. Chain-length agnostic; N is read from the input.

The naive form (review Sec. 7.1) has a hard ceiling, because there the odd
content enters only as sin(k*Delta_i) at the output bond, so v_j can depend
oddly on bonds j and j-1 alone -- worth 0.4555 against the GNN's 0.377 at every
size tried. Here an ODD channel stream is carried through the conv stack, so the
odd dependence has the same (unbounded) receptive field as the even one.

    Delta_i = wrap(theta_{i+1} - theta_i)                       [B, M], M = N-1
    e <- cos(k*Delta)   EVEN stream, convs WITH bias
    o <- sin(k*Delta)   ODD  stream, convs WITHOUT bias  (0 is the only fixed point)

Allowed couplings, by parity algebra: even*even -> even, odd*odd -> even,
even*odd -> odd, and any bias-free linear map preserves parity. So

    e <- e + conv_e( act( FiLM_t( [e, o * conv(o)] ) ) )        # odd^2 feeds even
    o <- o + conv_o(o) * gate(e)                                # even gates odd
    f <- sum_c o_c * head(act(e))_c                             [B, M]   ODD
    v_j = f_j - f_{j-1}                                         [B, N]   ODD, zero-sum

Palindromic kernels + zero padding make every step commute with reversing the
bond sequence, so v is site-reversal equivariant. Rotation invariance and the
2*pi-shift invariance are automatic (only wrapped bonds enter). At K=1, one head
channel with weight J and no conv layers this is the analytic score
J(sin D_j - sin D_{j-1}).
"""
from __future__ import annotations

import math

import torch
from torch import nn

def wrap(d):
    two_pi = 2.0 * math.pi
    return d - two_pi * torch.round(d / two_pi)


class PalindromicConv1d(nn.Module):
    """Conv1d whose kernel is symmetric in the spatial axis (w_j == w_{-j}).

    That is exactly the condition for the stack to commute with reversing the
    bond sequence, and it halves the kernel parameters.
    """

    def __init__(self, c_in, c_out, kernel=3, dilation=1):
        super().__init__()
        assert kernel % 2 == 1
        self.half = kernel // 2
        self.dilation = dilation
        # store only taps 0..half; tap 0 is the centre
        self.w = nn.Parameter(torch.empty(c_out, c_in, self.half + 1))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        self.b = nn.Parameter(torch.zeros(c_out))

    def kernel(self):
        # [c_out, c_in, 2*half+1], palindromic
        left = self.w[..., 1:].flip(-1)
        return torch.cat([left, self.w], dim=-1)

    def forward(self, h):                       # [B, C, M]
        k = self.kernel()
        pad = self.half * self.dilation
        return nn.functional.conv1d(h, k, self.b, padding=pad, dilation=self.dilation)




class _OddConv(PalindromicConv1d):
    """Palindromic conv with no bias -- required to preserve oddness."""

    def __init__(self, c_in, c_out, kernel=3, dilation=1):
        super().__init__(c_in, c_out, kernel, dilation)
        self.b = None

    def forward(self, h):
        k = self.kernel()
        pad = self.half * self.dilation
        return nn.functional.conv1d(h, k, None, padding=pad, dilation=self.dilation)


class XYParityNet(nn.Module):
    """Bond-space angle-flow network for the 1D XY chain. See the module docstring.

    Args:
        K: harmonics of the bond angle in each input stream (cos for even, sin
            for odd).
        C: even-stream channel width.
        Co: odd-stream channel width.
        n_blocks: residual blocks; `dilations` defaults to (1, 2, 4, ...).
        kernel: conv kernel size, odd. Stored palindromic, so (kernel+1)//2 free
            taps per (in, out) pair.
        dilations: explicit dilation per block, overriding `n_blocks`. Keep the
            receptive field near the correlation length (xi ~ 2.8 sites at J=2);
            reaching far past it costs chain-length generalisation.
        time_order: log-spaced frequency pairs appended to raw `t`; 0 is raw `t`.
        time_dim: width of the learned time embedding driving the per-block FiLM.
        act_fn: hidden activation (default SiLU).
    """

    def __init__(self, K=4, C=16, Co=8, n_blocks=2, kernel=3, dilations=None,
                 time_order=4, time_dim=12, act_fn=None):
        super().__init__()
        self.K, self.C, self.Co = K, C, Co
        self.time_order = time_order
        act = act_fn or nn.SiLU()
        self.act = act
        dil = dilations or tuple(2 ** i for i in range(n_blocks))
        self.dil = dil

        self.time_mlp = nn.Sequential(
            nn.Linear(1 + 2 * time_order, time_dim), act,
            nn.Linear(time_dim, time_dim), act)
        self.film = nn.Linear(time_dim, 2 * C * len(dil))

        self.in_e = PalindromicConv1d(K + 1, C, kernel=1)     # +1 end-indicator
        self.in_o = _OddConv(K, Co, kernel=1)
        self.pair = nn.ModuleList([_OddConv(Co, Co, kernel, d) for d in dil])
        self.conv_e = nn.ModuleList(
            [PalindromicConv1d(C + Co, C, kernel, d) for d in dil])
        self.conv_o = nn.ModuleList([_OddConv(Co, Co, kernel, d) for d in dil])
        self.gate = nn.ModuleList([PalindromicConv1d(C, Co, kernel=1) for _ in dil])
        self.head = PalindromicConv1d(C, Co, kernel=1)
        nn.init.zeros_(self.head.b)
        nn.init.normal_(self.head.w, std=1e-2)

    def _tfeat(self, t):
        if self.time_order < 1:
            return t.unsqueeze(-1)
        w = torch.logspace(0.0, math.log10(30.0), self.time_order,
                           device=t.device, dtype=t.dtype)
        a = t.unsqueeze(-1) * w
        return torch.cat([t.unsqueeze(-1), a.cos(), a.sin()], dim=-1)

    def forward(self, t, x):
        sq = x.dim() == 3 and x.shape[-1] == 1
        x2 = x.squeeze(-1) if sq else x
        B, N = x2.shape

        t = t if t.is_floating_point() else t.to(x2.dtype)
        t_b = t.expand(B) if t.dim() == 0 else t.reshape(B)
        g_t = self.time_mlp(self._tfeat(t_b))
        fs = self.film(g_t).view(B, len(self.dil), 2, self.C)

        D = wrap(x2[:, 1:] - x2[:, :-1])                          # [B, M]
        k = torch.arange(1, self.K + 1, device=D.device, dtype=D.dtype).view(1, -1, 1)
        a = D.unsqueeze(1) * k
        e = torch.cat([a.cos(), torch.ones_like(D).unsqueeze(1)], 1)   # EVEN
        o = a.sin()                                                    # ODD

        e = self.in_e(e)
        o = self.in_o(o)
        for i in range(len(self.dil)):
            inv = torch.cat([e, o * self.pair[i](o)], 1)               # EVEN
            sc, sh = fs[:, i, 0].unsqueeze(-1), fs[:, i, 1].unsqueeze(-1)
            inv = torch.cat([inv[:, :self.C] * (1.0 + sc) + sh, inv[:, self.C:]], 1)
            e = e + self.conv_e[i](self.act(inv))
            o = o + self.conv_o[i](o) * self.gate[i](self.act(e))      # ODD
        f = (o * self.head(self.act(e))).sum(1)                        # [B, M] ODD

        z = f.new_zeros(B, 1)
        v = torch.cat([f, z], 1) - torch.cat([z, f], 1)
        return v.unsqueeze(-1) if sq else v
