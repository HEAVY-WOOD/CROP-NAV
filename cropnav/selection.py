from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import CROPNavConfig


VIEW_NAMES = ("front", "left", "right", "rear", "down")


def sample_gumbel(logits):
    uniform = torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(uniform))


@dataclass
class ViewSelection:
    matrix: torch.Tensor  # (B, 3, 5), ordered horizontal choices followed by Down
    view_ids: torch.Tensor  # (B, 3), canonical camera indices
    logits: torch.Tensor  # (B, 4)

    def apply(self, tokens):
        """Route a tensor (B, 5, ...) using the same differentiable decision."""
        return torch.einsum("bsv,bv...->bs...", self.matrix.to(tokens), tokens)


class ViewSelector(nn.Module):
    """Two horizontal views without replacement, plus the mandatory Down view."""

    def __init__(self, cfg: CROPNavConfig):
        super().__init__()
        self.temperature = cfg.view_temperature
        self.summary_view_embedding = nn.Parameter(torch.randn(5, cfg.hidden_dim) * 0.02)
        self.output_view_embedding = nn.Parameter(torch.zeros(5, cfg.hidden_dim))
        self.summary_norm = nn.LayerNorm(cfg.hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.selector_dim), nn.GELU(),
            nn.Dropout(cfg.selector_dropout), nn.Linear(cfg.selector_dim, 1),
        )

    def forward(self, context_tokens, spatial_tokens):
        summary = context_tokens[:, :, 0] + spatial_tokens.mean(dim=2)
        summary = self.summary_norm(summary + self.summary_view_embedding)
        logits = self.scorer(summary).squeeze(-1)[:, :4]
        rows = []
        remaining = logits.float()
        # Both training and inference retain stochastic hard sampling.
        for _ in range(2):
            noisy = remaining + sample_gumbel(remaining)
            soft = F.softmax(noisy / self.temperature, dim=-1)
            hard = F.one_hot(noisy.argmax(-1), 4).to(soft)
            row = hard + (soft - soft.detach()) if self.training else hard
            rows.append(F.pad(row, (0, 1)))
            remaining = remaining.masked_fill(hard.bool(), float("-inf"))
        down = F.one_hot(torch.full_like(logits[:, 0], 4, dtype=torch.long), 5)
        matrix = torch.stack([*rows, down], dim=1).to(logits)
        return ViewSelection(matrix, matrix.detach().argmax(-1), logits)


@dataclass
class TokenSelection:
    tokens: tuple  # One (N_kept, D) tensor per sample; dense masked tokens in training
    keep_mask: torch.Tensor  # (B, N), sampled hard decisions
    keep_probabilities: torch.Tensor  # (B, N), sigmoid of the relevance logits
    retention_loss: torch.Tensor


class ContextTokenSelector(nn.Module):
    """Select spatial tokens using pooled goal and trajectory context."""

    def __init__(self, cfg: CROPNavConfig):
        super().__init__()
        self.temperature = cfg.token_temperature
        self.retention_weight = cfg.retention_weight
        self.token_proj = nn.Linear(cfg.hidden_dim, cfg.selector_dim)
        self.context_proj = nn.Linear(cfg.hidden_dim, cfg.selector_dim)
        self.score = nn.Linear(cfg.selector_dim, 1)

    def forward(self, spatial_tokens, context_tokens, trajectory_tokens, trajectory_mask=None):
        evidence = torch.cat([context_tokens, trajectory_tokens], dim=1)
        if trajectory_mask is None:
            context = evidence.mean(dim=1)
        else:
            mask = torch.cat([torch.ones(context_tokens.shape[:2], device=evidence.device,
                                         dtype=torch.bool), trajectory_mask], dim=1)
            context = (evidence * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        hidden = torch.tanh(self.token_proj(spatial_tokens) + self.context_proj(context).unsqueeze(1))
        logits = self.score(hidden).squeeze(-1)
        binary_logits = torch.stack([torch.zeros_like(logits), logits], dim=-1).float()
        soft = F.softmax((binary_logits + sample_gumbel(binary_logits)) / self.temperature, dim=-1)
        hard = F.one_hot(soft.argmax(-1), 2).to(soft)
        mask = (hard + (soft - soft.detach()) if self.training else hard)[..., 1]
        keep = mask.detach().bool()
        if self.training:
            tokens = tuple(spatial_tokens * mask.to(spatial_tokens).unsqueeze(-1))
        else:
            # Preserve spatial order; an all-drop decision produces an empty tensor.
            tokens = tuple(x[m] for x, m in zip(spatial_tokens, keep))
        probabilities = logits.sigmoid()
        return TokenSelection(tokens, keep, probabilities, probabilities.mean() * self.retention_weight)
