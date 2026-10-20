# Adapted from the Apache-2.0 trajectory implementation in the source project.
# See NOTICE for attribution and the scope of modifications.
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import CROPNavConfig


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc4(h) + self.skip(x)


class TrajectoryEncoder(nn.Module):
    """Encode motion direction, displacement, and relative time into LLM tokens."""

    def __init__(self, cfg: CROPNavConfig):
        super().__init__()
        self.cfg = cfg
        self.res_mlp = ResidualMLP(4 + cfg.time_embedding_dim, cfg.trajectory_hidden_dim, cfg.trajectory_dim)
        self.projection = nn.Linear(cfg.trajectory_dim, cfg.hidden_dim)

    def forward(self, waypoints):
        """waypoints: (B, T, 3), in the navigation policy's coordinate frame."""
        if waypoints.ndim != 3 or waypoints.shape[-1] != 3:
            raise ValueError("waypoints must have shape (B, T, 3)")
        waypoints = waypoints.to(self.res_mlp.fc1.weight)[:, -self.cfg.max_history:]
        b, t, _ = waypoints.shape
        # A repeated initial position represents zero motion at episode start.
        if t == 0:
            waypoints = waypoints.new_zeros(b, 2, 3)
        elif t == 1:
            waypoints = waypoints.expand(-1, 2, -1)
        delta = waypoints[:, 1:] - waypoints[:, :-1]
        distance = delta.norm(dim=-1, keepdim=True)
        motion = torch.cat([delta / (distance + self.cfg.epsilon), distance], dim=-1)
        dim = self.cfg.time_embedding_dim
        positions = torch.arange(delta.shape[1], device=delta.device, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, dim, 2, device=delta.device) * (-math.log(10000.0) / dim))
        phase = positions * scale
        time = torch.zeros(delta.shape[1], dim, device=delta.device)
        time[:, 0::2] = phase.sin()
        time[:, 1::2] = phase[:, :dim // 2].cos()
        features = torch.cat([motion, time.to(motion).expand(b, -1, -1)], dim=-1)
        return self.projection(self.res_mlp(features))
