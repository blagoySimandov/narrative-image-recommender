## type: ignore
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from models import DataModel

IMAGES_ROOT = Path("images")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model(json_path: Path) -> DataModel:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DataModel(**data)


def find_album_images(album_id: str) -> list:
    album_dir = IMAGES_ROOT / album_id
    if not album_dir.exists():
        return []
    return sorted(album_dir.glob("*.jpg"))


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
