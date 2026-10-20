# Copyright 2023 Haotian Liu
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
# The waypoint-head structure is adapted from the source project; prompt packing
# and the modular CROP-NAV policy interface have been rewritten for this release.
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .model import CROPNav, CROPNavOutput


@dataclass(frozen=True)
class PromptMarkers:
    """Sentinels in tokenized prompts, replaced before embedding lookup."""

    current: int = -200
    memory: int = -201
    trajectory: int = -202
    action: int = -203

    def __post_init__(self):
        if len({self.current, self.memory, self.trajectory, self.action}) != 4:
            raise ValueError("Prompt markers must be distinct")


class WaypointHead(nn.Module):
    """Map the action-query hidden state to (direction_x, direction_y, direction_z, distance)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64),
        )
        self.output = nn.Linear(64, 4)

    def forward(self, hidden):
        return self.output(self.features(hidden))


@dataclass
class NavigationOutput:
    waypoint: torch.Tensor  # (B, 4), raw direction and distance predictions
    compression: CROPNavOutput


class CROPNavPolicy(nn.Module):
    """Insert compact evidence into a causal language decoder and predict a waypoint.

    decoder must expose get_input_embeddings() and return last_hidden_state.
    The decoder, waypoint head, and action embedding require matching weights.
    """

    def __init__(self, compression: CROPNav, decoder: nn.Module, markers=None,
                 waypoint_head=None, action_embedding=None):
        super().__init__()
        self.compression = compression
        self.decoder = decoder
        self.markers = markers if markers is not None else PromptMarkers()
        d = compression.config.hidden_dim
        if decoder.get_input_embeddings().weight.shape[-1] != d:
            raise ValueError("Decoder and CROP-NAV hidden dimensions must match")
        self.waypoint_head = waypoint_head if waypoint_head is not None else WaypointHead(d)
        self.action_embedding = nn.Parameter(torch.empty(1, d))
        nn.init.normal_(self.action_embedding, std=0.02)
        if action_embedding is not None:
            if action_embedding.numel() != d:
                raise ValueError("action_embedding must contain one hidden-size vector")
            with torch.no_grad():
                self.action_embedding.copy_(action_embedding.reshape(1, d))

    def _pack(self, input_ids, compression, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != len(compression.current_tokens):
            raise ValueError("input_ids must have shape (B, L) matching the visual batch")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        embedding = self.decoder.get_input_embeddings()
        sequences, action_positions = [], []
        marker_ids = (self.markers.current, self.markers.memory, self.markers.trajectory, self.markers.action)
        for index, ids in enumerate(input_ids):
            ids = ids[attention_mask[index].bool()].to(embedding.weight.device)
            positions = []
            for marker in marker_ids:
                location = (ids == marker).nonzero(as_tuple=True)[0]
                if location.numel() != 1:
                    raise ValueError("Each prompt must contain each of the four markers exactly once")
                positions.append(int(location.item()))
            if positions[-1] < max(positions[:-1]):
                raise ValueError("The action query must follow all visual and trajectory markers")
            replacements = dict(zip(marker_ids, (
                compression.current_tokens[index], compression.memory_tokens[index],
                compression.trajectory_tokens[index], self.action_embedding,
            )))
            pieces, offset, length = [], 0, 0
            for position in sorted(positions):
                text_ids = ids[offset:position]
                if text_ids.numel():
                    pieces.append(embedding(text_ids))
                    length += text_ids.numel()
                marker = int(ids[position].item())
                if marker == self.markers.action:
                    action_positions.append(length)
                replacement = replacements[marker].to(embedding.weight)
                pieces.append(replacement)
                length += replacement.shape[0]
                offset = position + 1
            if offset < ids.numel():
                pieces.append(embedding(ids[offset:]))
            sequences.append(torch.cat(pieces, dim=0))
        inputs = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(x) for x in sequences], device=inputs.device)
        mask = torch.arange(inputs.shape[1], device=inputs.device)[None] < lengths[:, None]
        position_ids = mask.long().cumsum(-1).sub(1).clamp_min(0)
        return inputs, mask, position_ids, torch.tensor(action_positions, device=inputs.device)

    def forward(self, context_tokens, spatial_tokens, waypoints, input_ids,
                memory=None, attention_mask=None, waypoint_lengths=None):
        compressed = self.compression(context_tokens, spatial_tokens, waypoints, memory, waypoint_lengths)
        return NavigationOutput(self.decode(input_ids, compressed, attention_mask), compressed)

    def decode(self, input_ids, compressed, attention_mask=None):
        """Predict from compressed evidence; trajectory rows may have different lengths."""
        inputs, mask, position_ids, positions = self._pack(input_ids, compressed, attention_mask)
        hidden = self.decoder(
            inputs_embeds=inputs, attention_mask=mask, position_ids=position_ids,
            use_cache=False, return_dict=True,
        ).last_hidden_state
        action = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return self.waypoint_head(action.to(self.action_embedding))
