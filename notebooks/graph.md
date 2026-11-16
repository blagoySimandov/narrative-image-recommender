---
title: Graph
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import polars as pl
import numpy as np
from torch import threading
from dataset_loaders import load_yfcc_city
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
```

Load the amsterdam dataset

```python {.marimo}
amsterdam = load_yfcc_city("amsterdam")
amsterdam
```

Embed all images

```python {.marimo}
from transformers import pipeline

dino_pipe = pipeline(task="image-feature-extraction", model="facebook/dinov2-small")

def add_dino_embeddings(
    df: pl.DataFrame, col: str, batch_size: int = 32
) -> pl.DataFrame:
    out = dino_pipe(df[col].to_list(), pool=True, batch_size=batch_size)
    emb = np.asarray(out, dtype=np.float32).reshape(len(df), -1)
    return df.with_columns(pl.Series("dino_embeddings", emb))
```

```python {.marimo}
embed = add_dino_embeddings(amsterdam,"path")
```

Run musiq to get an aesthetic score

```python {.marimo}
musiq = hub.load("https://www.kaggle.com/models/google/musiq/tensorFlow2/koniq-10k/1")
musiq_fn = musiq.signatures["serving_default"]

def musiq_score(path: str) -> float:
    img_bytes = tf.constant(Path(path).read_bytes())
    return float(list(musiq_fn(img_bytes).values())[0].numpy())

def add_musiq_scores(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col).map_elements(musiq_score, return_dtype=pl.Float64).alias("musiq_score")
# .apply(musiq_score).alias("musiq_score")
        )
```

```python {.marimo}
X = add_musiq_scores(embed,"path")
X
```

Create a graph.
Edges are cos(a,b)
Nodes are Aesthetic scores as prize

Run TSP prize collection
Find the best path with length 10

Edges are between low and high thresholds
and for each ij if datetaken [i] < datetaken [j]

```python {.marimo}
from networkx import Graph
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

G = Graph()
for i in range(len(X)):
    G.add_node(i, aesthetic=X["musiq_score"][i])
threasholdh = 0.6
threasholdl = 0.5
similarity_matrix = cosine_similarity(np.stack(X["dino_embeddings"].to_numpy())) # type: ignore

for i in range(len(X)):
    for j in range(i+1, len(X)):
        weight = similarity_matrix[i,j]
        if threasholdl<weight<threasholdh and X['datetaken'][i] < X['datetaken'][j]:
            G.add_edge(i, j, weight=weight)
```

```python {.marimo}
# Option A: neighbourhood of one photo
H = G
```

Draw the graph

```python {.marimo}
import base64, io
from PIL import Image
from pyvis.network import Network

def thumb_uri(path, size=120):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# shuffled grid as starting positions
rng = np.random.default_rng(0)
nodes = [_n for _n in G.nodes if G.degree(_n) > 0]
rng.shuffle(nodes)
cols = int(np.ceil(np.sqrt(len(nodes))))
spacing = 250

net = Network(height="850px", width="100%", cdn_resources="remote", directed=G.is_directed())
for _i, _n in enumerate(nodes):
    net.add_node(
        int(_n),
        shape="image",
        image=thumb_uri(X["path"][_n]),
        size=35,
        title=f"#{_n} | MUSIQ {G.nodes[_n]['aesthetic']:.1f}",
        x=(_i % cols) * spacing,
        y=(_i // cols) * spacing,
    )

for _u, _v, _w in G.edges(data="weight"):
    net.add_edge(int(_u), int(_v), value=float(_w), label=f"{_w:.2f}", title=f"{_w:.2f}")

net.toggle_physics(False)  # keep the grid; set True to let similar photos pull together
net.set_edge_smooth("dynamic")

mo.iframe(net.generate_html(), height="870px")
```

```python {.marimo}

```
