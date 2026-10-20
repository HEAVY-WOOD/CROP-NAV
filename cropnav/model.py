from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import CROPNavConfig
from .memory import DirectionalMemory
from .selection import ContextTokenSelector, ViewSelector
from .trajectory import TrajectoryEncoder


@dataclass
class CROPNavOutput:
    current_tokens: tuple
    trajectory_tokens: torch.Tensor | tuple
    memory_tokens: torch.Tensor
    memory: torch.Tensor
    selected_view_ids: torch.Tensor
    token_keep_mask: torch.Tensor
    token_keep_probabilities: torch.Tensor
    retention_loss: torch.Tensor


class CROPNav(nn.Module):
    """The complete coarse-to-fine compression and recurrent context pipeline."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else CROPNavConfig()
        self.view_selector = ViewSelector(self.config)
        self.token_selector = ContextTokenSelector(self.config)
        self.memory = DirectionalMemory(self.config)
        self.trajectory = TrajectoryEncoder(self.config)
        self.memory_projection = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)

    def forward(self, context_tokens, spatial_tokens, waypoints, memory=None, waypoint_lengths=None):
        """One online navigation step; state is explicit and belongs to the caller.

        context_tokens: (B, 5, 1, D), goal-conditioned context per camera.
        spatial_tokens: (B, 5, N, D), complete projected spatial features.
        waypoints: (B, T, 3); memory: optional (B, 5, K, D).
        """
        if spatial_tokens.ndim != 4 or spatial_tokens.shape[1] != 5:
            raise ValueError("spatial_tokens must have shape (B, 5, N, D)")
        b, _, n, d = spatial_tokens.shape
        if n < self.config.instability_top_r or d != self.config.hidden_dim:
            raise ValueError("Spatial token count or hidden dimension does not match config")
        if context_tokens.shape != (b, 5, 1, d) or waypoints.shape[0] != b:
            raise ValueError("Context, spatial features, and waypoints must have matching batches")
        param = self.view_selector.summary_view_embedding
        context_tokens, spatial_tokens = context_tokens.to(param), spatial_tokens.to(param)
        if memory is not None:
            memory = memory.to(param)
        selection = self.view_selector(context_tokens, spatial_tokens)
        selected_context = selection.apply(context_tokens)
        selected_spatial = selection.apply(spatial_tokens)
        # Memory always receives the complete spatial sets, before token pruning.
        state = self.memory(selected_spatial, selection.matrix, memory)
        embeddings = torch.einsum("bsv,vd->bsd", selection.matrix, self.view_selector.output_view_embedding)
        selected_context = selected_context + embeddings.unsqueeze(2)
        selected_spatial = selected_spatial + embeddings.unsqueeze(2)
        trajectory = self.trajectory(waypoints)
        trajectory_mask = None
        if waypoint_lengths is not None:
            lengths = waypoint_lengths.clamp(min=2, max=self.config.max_history) - 1
            trajectory_mask = torch.arange(trajectory.shape[1], device=trajectory.device)[None] < lengths[:, None]
        context = selected_context.flatten(1, 2)
        fine = self.token_selector(selected_spatial.flatten(1, 2), context, trajectory, trajectory_mask)
        current = tuple(torch.cat([q, x], dim=0) for q, x in zip(context, fine.tokens))
        memory_tokens = self.memory_projection(state) + self.view_selector.output_view_embedding[None, :, None]
        if trajectory_mask is not None:
            trajectory = tuple(row[mask] for row, mask in zip(trajectory, trajectory_mask))
        return CROPNavOutput(
            current, trajectory, memory_tokens.flatten(1, 2), state,
            selection.view_ids, fine.keep_mask, fine.keep_probabilities, fine.retention_loss,
        )

    def warmup_memory(self, spatial_history, memory=None, valid_mask=None):
        """Reconstruct memory from (B, T, 5, N, D), with coarse selection bypassed."""
        if spatial_history.ndim != 5 or spatial_history.shape[2] != 5:
            raise ValueError("spatial_history must have shape (B, T, 5, N, D)")
        spatial_history = spatial_history.to(self.memory.slot_init)
        b = spatial_history.shape[0]
        state = self.memory.initial_state(b) if memory is None else memory.to(spatial_history)
        identity = torch.eye(5, device=state.device, dtype=state.dtype).expand(b, -1, -1)
        for index, frame in enumerate(spatial_history.unbind(dim=1)):
            updated = self.memory(frame, identity, state)
            state = updated if valid_mask is None else torch.where(valid_mask[:, index, None, None, None], updated, state)
        return state

    def save_pretrained(self, directory):
        """Save only CROP-NAV modules, separately from supplied pretrained backbones."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "config.json")
        torch.save(self.state_dict(), directory / "model.pt")

    @classmethod
    def from_pretrained(cls, directory, map_location="cpu"):
        directory = Path(directory)
        model = cls(CROPNavConfig.load(directory / "config.json"))
        weights = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(weights, strict=True)
        return model.to(map_location).eval()
