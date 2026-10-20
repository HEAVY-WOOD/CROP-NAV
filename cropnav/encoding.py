# The context-attention assembly is adapted from the Apache-2.0 source project.
# See NOTICE. Pretrained encoder implementations and weights are supplied externally.
import math

import torch
from torch import nn
from torch.nn import functional as F


class GoalConditionedEncoder(nn.Module):
    """EVA/Q-Former adapter producing one context token and an 8x8 spatial grid.

    All constructor modules must already be initialized with compatible weights.
    Q-Former uses the BLIP-style `.bert(..., query_embeds=...)` interface.
    """

    def __init__(self, vision_tower, qformer, tokenizer, query_tokens,
                 query_projection, key_projection, context_projection, spatial_projection,
                 vision_norm=None, vision_projection=None, grid_size=8):
        super().__init__()
        self.vision_tower = vision_tower
        self.qformer = qformer
        self.tokenizer = tokenizer
        self.query_tokens = query_tokens if isinstance(query_tokens, nn.Parameter) else nn.Parameter(query_tokens)
        self.query_projection = query_projection
        self.key_projection = key_projection
        self.context_projection = context_projection
        self.spatial_projection = spatial_projection
        self.vision_norm = vision_norm if vision_norm is not None else nn.Identity()
        self.vision_projection = vision_projection if vision_projection is not None else nn.Identity()
        self.grid_size = grid_size

    def forward(self, images, instructions):
        """images: preprocessed (B, 5, 3, H, W); instructions: B goal strings."""
        if images.ndim != 5 or images.shape[1:3] != (5, 3) or len(instructions) != images.shape[0]:
            raise ValueError("Expected five RGB views and one instruction per sample")
        b = images.shape[0]
        vision_param = next(self.vision_tower.parameters())
        features = self.vision_tower(images.flatten(0, 1).to(vision_param))
        if features.ndim != 3:
            raise ValueError("vision_tower must return patch features (B*5, N, C)")
        q_features = self.vision_norm(self.vision_projection(features))
        text = self.tokenizer(
            [goal for goal in instructions for _ in range(5)],
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        )
        ids = text["input_ids"].to(features.device)
        text_mask = text["attention_mask"].to(features.device)
        queries = self.query_tokens.expand(b * 5, -1, -1)
        query_mask = torch.ones(queries.shape[:2], device=ids.device, dtype=text_mask.dtype)
        output = self.qformer.bert(
            ids, query_embeds=queries,
            attention_mask=torch.cat([query_mask, text_mask], dim=1),
            encoder_hidden_states=q_features,
            encoder_attention_mask=torch.ones(q_features.shape[:2], device=ids.device, dtype=torch.long),
            return_dict=True,
        ).last_hidden_state[:, :queries.shape[1]]
        queries = self.query_projection(output)
        count = features.shape[1]
        # EVA may expose its CLS token; Q-Former sees it, spatial attention excludes it.
        side = math.isqrt(count)
        patches = features if side * side == count else features[:, 1:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("Vision patch features must form a square spatial grid")
        scores = (queries @ self.key_projection(patches).transpose(-1, -2)) / math.sqrt(patches.shape[-1])
        context = (scores.float().softmax(-1).to(patches) @ patches).mean(1, keepdim=True)
        context = self.context_projection(context)
        grid = patches.transpose(1, 2).reshape(b * 5, -1, side, side)
        spatial = F.adaptive_avg_pool2d(grid, self.grid_size).flatten(2).transpose(1, 2)
        spatial = self.spatial_projection(spatial)
        return context.reshape(b, 5, 1, -1), spatial.reshape(b, 5, self.grid_size ** 2, -1)
