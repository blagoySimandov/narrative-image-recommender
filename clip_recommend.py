# %% [markdown]
# # CLIP Image Recommender
#
# This notebook takes a text query. It finds the images in an album that match
# the query. It uses CLIP to compare the text and the images.
#
# ## References
#
# - CLIP paper: https://arxiv.org/abs/2103.00020
# - VIST dataset paper: https://arxiv.org/abs/1604.03968
#
# Page 4, Section 2.3
#
# > "multi-modal embedding space by jointly training an image encoder and
# > text encoder to maximize the cosine similarity of the image and text
# > embeddings of the N real pairs in the batch while minimizing the cosine
# > similarity of the embeddings of the N^2 minus N incorrect pairings."
#
# Page 5, Figure 3 of the CLIP paper gives the pseudocode. The functions
# `encode_images`, `encode_text`, and `rank_images` below follow this
# pseudocode: normalize each embedding, then take the dot product to get the
# cosine similarity score.

# %%
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from models import DataModel

IMAGES_ROOT = Path("images")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


# %% [markdown]
# ## Load the Dataset
#
# This function reads the SIS json file. It gives back a `DataModel` object
# with the images, the albums, and the story text.

# %%
def load_model(json_path: Path) -> DataModel:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DataModel(**data)


# %% [markdown]
# ## Find Album Images
#
# This function finds the downloaded images for one album. It looks in the
# `images` folder, under a subfolder with the album id.

# %%
def find_album_images(album_id: str) -> list:
    album_dir = IMAGES_ROOT / album_id
    if not album_dir.exists():
        return []
    return sorted(album_dir.glob("*.jpg"))


# %% [markdown]
# ## Encode the Images
#
# This function turns each image into a vector with CLIP. The vector is a
# list of numbers. CLIP puts images and text in the same vector space, so you
# can compare them.
#
# The function also normalizes each vector. Normalization sets the length of
# the vector to 1. This step makes the similarity score correct.

# %%
def encode_images(image_paths: list, processor, model, device: str) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


# %% [markdown]
# ## Encode the Text Query
#
# This function turns the text query into a vector with CLIP. It uses the
# same vector space as the image function above.

# %%
def encode_text(query: str, processor, model, device: str) -> torch.Tensor:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


# %% [markdown]
# ## Rank the Images
#
# This function compares the text vector to each image vector. It computes a
# score for each image. A higher score means a closer match to the text.
#
# The function then gives back the top matches, in order from the highest
# score to the lowest score.

# %%
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


# %% [markdown]
# ## Set the Inputs
#
# Set the json path, the album id, the text query, and the top k value here.
# Change these values to try a different album or a different query.

# %%
json_path = Path("sis/test.story-in-sequence.json")
album_id = "504823"
query = "cat going to sleep"
top_k = 5

# %% [markdown]
# ## Load the CLIP Model
#
# This step loads the CLIP model and the processor. The processor prepares
# the images and the text for the model. This step can take a moment on the
# first run, since it downloads the model weights.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
model.eval()

# %% [markdown]
# ## Find the Images
#
# This step finds the downloaded images for the album set above.

# %%
image_paths = find_album_images(album_id)
image_paths

# %% [markdown]
# ## Run the Encoders
#
# This step encodes the images and the text query with CLIP.

# %%
image_features = encode_images(image_paths, processor, model, device)
text_feature = encode_text(query, processor, model, device)

# %% [markdown]
# ## Show the Top Matches
#
# This step ranks the images against the query and plots the top matches.

# %%
import matplotlib.pyplot as plt

results = rank_images(image_paths, image_features, text_feature, top_k)

fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
for ax, (path, score) in zip(axes, results):
    ax.imshow(Image.open(path))
    ax.set_title(f"{score:.4f}\n{path.name}")
    ax.axis("off")
plt.show()


# %%
