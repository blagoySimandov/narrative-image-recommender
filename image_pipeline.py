from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pillow_heif import register_heif_opener
from transformers import CLIPModel, CLIPProcessor

register_heif_opener()

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


@dataclass
class ImageRecord:
    path: Path
    id: str


@dataclass
class ImageEntry:
    path: Path
    embedding: object
    caption: str


class ImageBase:
    def __init__(self, entries: list):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, index: int):
        return self.entries[index]

    @property
    def paths(self) -> list:
        return [e.path for e in self.entries]

    @property
    def embeddings(self) -> torch.Tensor:
        return torch.stack([e.embedding for e in self.entries])

    @property
    def captions(self) -> dict:
        return {e.path: e.caption for e in self.entries}

    def remove(self, path: Path) -> None:
        self.entries = [e for e in self.entries if e.path != path]


def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()
    return processor, model, device


def encode_images(
    image_paths: list, processor: CLIPProcessor, model, device: str
) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


def encode_text(query: str, processor, model, device: str) -> torch.Tensor:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


def rank_images(
    image_paths: list,
    image_features: torch.Tensor,
    text_feature: torch.Tensor,
    top_k: int,
):
    scores = (image_features @ text_feature.T).squeeze(1)
    topk = scores.topk(min(top_k, len(image_paths)))
    return [
        (image_paths[i], float(s))
        for i, s in zip(topk.indices.tolist(), topk.values.tolist())
    ]


def story_strip(mo, story_rows):
    """mo is passed in so this module has no hard marimo dependency."""

    def story_card(path, caption):
        return mo.vstack([
            mo.image(src=str(path), width=220),
            mo.md(f"**{caption}**"),
        ], align="center")

    cards = []
    for i, row in enumerate(story_rows):
        cards.append(story_card(row["path"], row["caption"]))
        if i < len(story_rows) - 1:
            cards.append(mo.md("### →"))

    return mo.hstack(cards, align="center", gap=1, wrap=True)


def plot_results_grid(results_by_query: dict, top_k: int):
    fig, axes = plt.subplots(
        len(results_by_query), top_k, figsize=(4 * top_k, 4 * len(results_by_query))
    )
    for row, (q, results) in enumerate(results_by_query.items()):
        for col, (path, score) in enumerate(results):
            ax = axes[row, col]
            ax.imshow(Image.open(path))
            ax.set_title(f"{score:.4f}\n{path.name}")
            ax.axis("off")
        axes[row, 0].set_ylabel(q, rotation=0, labelpad=60, fontsize=12)
    plt.tight_layout()
    plt.show()
