import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import CROPNavConfig


@torch.no_grad()
def instability_multipliers(tokens, top_r=1, sigma=64.0, strength=0.5, epsilon=1e-6):
    """Channel-frequency residual voting, independently per selected view.

    Input (..., N, D); output (..., N), with positive, unit-mean multipliers.
    All N spatial tokens remain available for memory attention.
    """
    n, d = tokens.shape[-2:]
    if not 1 <= top_r <= n:
        raise ValueError("top_r must be between one and the number of spatial tokens")
    if sigma <= 0 or epsilon <= 0 or not 0 <= strength < 1:
        raise ValueError("Invalid instability parameters")
    x = tokens.float()
    frequencies = torch.fft.fftfreq(d, device=x.device) * d
    gaussian = torch.exp(-0.5 * (frequencies / sigma).square())
    lowpass = torch.fft.ifft(torch.fft.fft(x, dim=-1) * gaussian, dim=-1).real
    residual = (x - lowpass).abs() / (x.abs() + epsilon)
    indices = residual.topk(top_r, dim=-2).indices
    votes = torch.zeros_like(x[..., 0])
    votes.scatter_add_(-1, indices.flatten(-2), torch.ones_like(indices.flatten(-2), dtype=x.dtype))
    return (1 - strength) + strength * n * votes / (top_r * d)


class DirectionalMemory(nn.Module):
    """IGTA attention and GRU updates in a persistent five-direction slot bank."""

    def __init__(self, cfg: CROPNavConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.slot_init = nn.Parameter(torch.randn(cfg.num_slots, d) * 0.02)
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)

    def initial_state(self, batch_size):
        return self.slot_init[None, None].expand(batch_size, 5, -1, -1).clone()

    def _update(self, delta, previous):
        if not (delta.is_cuda and delta.dtype == torch.bfloat16):
            return self.gru(delta, previous)
        # Avoid unsupported fused BF16 GRU kernels without mutating parameter dtype.
        gi = F.linear(delta, self.gru.weight_ih, self.gru.bias_ih).float()
        gh = F.linear(previous, self.gru.weight_hh, self.gru.bias_hh).float()
        ir, iz, inn = gi.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
        candidate = (inn + reset * hn).tanh()
        return (candidate + update * (previous.float() - candidate)).to(delta.dtype)

    def forward(self, spatial_tokens, selection_matrix, state=None):
        """Write only routed directions; return all five rows.

        spatial_tokens: (B, S, N, D), S=3 online or S=5 for history warm-up.
        selection_matrix: (B, S, 5); state: optional (B, 5, K, D).
        """
        b, _, _, d = spatial_tokens.shape
        if state is None:
            state = self.initial_state(b)
        if state.shape != (b, 5, self.cfg.num_slots, self.cfg.hidden_dim):
            raise ValueError("Memory must have shape (B, 5, num_slots, hidden_dim)")
        p = selection_matrix.to(spatial_tokens)
        previous = torch.einsum("bsv,bvkd->bskd", p, state)
        multipliers = instability_multipliers(
            spatial_tokens, self.cfg.instability_top_r, self.cfg.instability_sigma,
            self.cfg.instability_strength, self.cfg.epsilon,
        )
        q, k, v = self.w_q(previous), self.w_k(spatial_tokens), self.w_v(spatial_tokens)
        logits = torch.einsum("bskd,bsnd->bskn", q, k).float() / math.sqrt(d)
        attention = (logits + multipliers.log().unsqueeze(-2)).softmax(dim=-1).to(v)
        delta = torch.einsum("bskn,bsnd->bskd", attention, v)
        updated = self._update(delta.reshape(-1, d), previous.reshape(-1, d)).reshape_as(previous)
        write_mask = p.sum(dim=1)[..., None, None]
        return state * (1 - write_mask) + torch.einsum("bsv,bskd->bvkd", p, updated)
