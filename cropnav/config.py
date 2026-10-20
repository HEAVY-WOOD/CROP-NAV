from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CROPNavConfig:
    """Dimensions and hyperparameters of the paper's complete method."""

    hidden_dim: int = 4096
    selector_dim: int = 256
    selector_dropout: float = 0.1
    view_temperature: float = 0.7
    token_temperature: float = 1.0
    num_slots: int = 32
    trajectory_dim: int = 512
    trajectory_hidden_dim: int = 256
    time_embedding_dim: int = 16
    max_history: int = 64
    instability_top_r: int = 1
    instability_sigma: float = 64.0
    instability_strength: float = 0.5
    epsilon: float = 1e-6
    retention_weight: float = 5e-5

    def __post_init__(self):
        dimensions = (
            self.hidden_dim, self.selector_dim, self.num_slots,
            self.trajectory_dim, self.trajectory_hidden_dim,
            self.time_embedding_dim, self.max_history, self.instability_top_r,
        )
        if any(d <= 0 for d in dimensions):
            raise ValueError("All dimensions, history length, and top_r must be positive")
        if min(self.view_temperature, self.token_temperature, self.instability_sigma, self.epsilon) <= 0:
            raise ValueError("Temperatures, Gaussian sigma, and epsilon must be positive")
        if not 0 <= self.instability_strength < 1:
            raise ValueError("instability_strength must be in [0, 1)")
        if not 0 <= self.selector_dropout < 1 or self.retention_weight < 0:
            raise ValueError("Invalid dropout or retention weight")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
