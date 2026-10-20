"""Five-view navigation samples from trajectory JSON and image archives."""
import io
import json
from pathlib import Path
import tarfile

import torch
from torch.utils.data import Dataset

from .selection import VIEW_NAMES


class NavigationDataset(Dataset):
    """Index entries: {"json": "scene/episode/merged_data.json", "frame": 6}.

    Frames in the index are one-based. Images in the trajectory archive are
    zero-based and ordered Front, Left, Right, Rear, Down. Only past observations
    and the current observation are returned; the next position supplies labels.
    """

    def __init__(self, index_path, dataset_root, image_processor, history_frames=5):
        self.root = Path(dataset_root)
        self.entries = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.processor = image_processor
        self.history_frames = history_frames
        if not self.entries:
            raise ValueError("The training index is empty")

    def __len__(self):
        return len(self.entries)

    def _images(self, directory, start, end):
        tensor_file = directory / "rgb_imgs.tensor"
        if tensor_file.is_file():
            images = torch.load(tensor_file, map_location="cpu", weights_only=True)
            return images[start * 5:end * 5].reshape(end - start, 5, 3, *images.shape[-2:]).float()
        from PIL import Image
        images = []
        with tarfile.open(directory / "rgb_imgs.tar", "r") as archive:
            for frame in range(start, end):
                for view in VIEW_NAMES:
                    name = f"{frame:06d}.{view}camera.jpg"
                    stream = archive.extractfile(name)
                    if stream is None:
                        raise ValueError(f"Missing image {name}")
                    images.append(Image.open(io.BytesIO(stream.read())).convert("RGB"))
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return pixels.reshape(end - start, 5, *pixels.shape[1:])

    def __getitem__(self, index):
        item = self.entries[index]
        path = self.root / item["json"]
        sample = json.loads(path.read_text(encoding="utf-8"))
        positions = torch.tensor(sample["trajectory"], dtype=torch.float32)[:, :3]
        frame = int(item["frame"])
        if not 1 <= frame <= len(positions):
            raise ValueError(f"Invalid one-based frame {frame} in {path}")
        # Match the navigation coordinate convention used by the waypoint labels.
        axis = positions[-1, :2]
        length = axis.norm()
        axis = axis / length if length > 1e-8 else axis.new_tensor([1., 0.])
        rotation = torch.eye(3)
        rotation[:2, 0] = axis
        rotation[:2, 1] = torch.stack([-axis[1], axis[0]])
        history = positions[:frame] @ rotation
        delta = (positions[min(frame, len(positions) - 1)] - positions[frame - 1]) @ rotation
        distance = delta.norm()
        target = torch.cat([delta / (distance + 1e-6), distance[None]])
        instruction = next(turn["value"] for turn in sample["conversations"] if turn["from"] == "human")
        instruction = instruction.replace("<image>", "").strip()
        start = max(0, frame - 1 - self.history_frames)
        return {
            "images": self._images(path.parent, start, frame),
            "instruction": instruction, "waypoints": history,
            "target": target, "sample_id": f"{item['json']}:{frame}",
        }


def navigation_prompt(tokenizer, instruction, markers):
    """Tokenize language segments and insert non-vocabulary evidence markers."""
    parts = [
        (f"USER: Instruction: {instruction}\nHistory images: ", markers.memory),
        ("\nHistory waypoints: ", markers.trajectory),
        ("\nCurrent image: ", markers.current),
        ("\nPredict the next waypoint.\nASSISTANT: ", markers.action),
    ]
    ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    for text, marker in parts:
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        ids.append(marker)
    return torch.tensor([ids], dtype=torch.long)
