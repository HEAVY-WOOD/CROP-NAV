# Model interface

## Encoded observations

`CROPNav.forward(context_tokens, spatial_tokens, waypoints, memory=None)` accepts:

| Input | Shape |
| --- | --- |
| Goal-conditioned context tokens | `(B, 5, 1, D)` |
| Spatial tokens | `(B, 5, N, D)` |
| Waypoint history | `(B, T, 3)` |
| Recurrent memory | `(B, 5, K, D)` or `None` |

Camera order is **Front, Left, Right, Rear, Down**. Defaults are `D=4096`, `N=64`,
and `K=32`. The trajectory encoder retains up to 64 positions.

The output contains compact current tokens, trajectory tokens, all `5K` memory
tokens, the next memory state, selected view IDs, token masks, and retention loss.
Current tokens retain the three context tokens followed by selected spatial tokens.
Both selectors retain stochastic Gumbel sampling during inference.

## Online inference

`GoalConditionedEncoder` accepts preprocessed `(B, 5, 3, H, W)` images and one
instruction per sample. `CROPNavPolicy` assembles the multimodal sequence and
predicts a raw 3D direction and distance.

```python
import torch

# encoder and policy are initialized with matching pretrained weights.
memory = None
encoder.eval()
policy.eval()
with torch.no_grad():
    context, spatial = encoder(images, instructions)
    result = policy(context, spatial, waypoints, prompt_ids, memory=memory)
    memory = result.compression.memory
    waypoint = result.waypoint
```

Carry the returned memory into the next step; reset it to `None` for a new
navigation episode. For a preceding observation window,
`model.warmup_memory(spatial_history)` accepts `(B, T, 5, N, D)` and writes all
five directions. Online updates write only the three selected directions.

## Decoder and prompt

The decoder exposes `get_input_embeddings()`, accepts `inputs_embeds`,
`attention_mask`, and `position_ids`, and returns `.last_hidden_state`.
The prompt contains each marker once, with the action query after the others:

| Marker | Default ID | Replacement |
| --- | --- | --- |
| Memory | `-201` | All five updated memory rows |
| Trajectory | `-202` | Encoded waypoint history |
| Current | `-200` | Selected context and spatial tokens |
| Action | `-203` | Learned action query |

Use `PromptMarkers` to match existing token IDs. These placeholders are replaced
before vocabulary lookup. `navigation_prompt` in `cropnav.data` builds this layout.

`CROPNav.save_pretrained(path)` saves the method modules and configuration.
The pretrained encoder, decoder, action query, and waypoint head require their
corresponding weights; training checkpoints include the method/head parameters.
