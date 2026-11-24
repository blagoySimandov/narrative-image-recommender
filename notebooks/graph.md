---
title: Graph
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from dataset_loaders import load_yfcc_city
from evaluation import comparison_chart, normalise, score_systems, summarise
from features import add_dino_embeddings, add_musiq_scores
from systems.graph_path import best_budget_path, build_graph
```

The code for this system is in `systems/graph_path.py`.
The code for the eval is in `evaluation.py`.
Use `notebooks/compare.md` to compare this system with others.

Load the amsterdam dataset

```python {.marimo}
city_name = "amsterdam"
amsterdam = load_yfcc_city(city_name)
amsterdam
```

Embed all images with DINOv2 and score them with MUSIQ (`ava`)

```python {.marimo}
with mo.persistent_cache(name=f"features-{city_name}-ava"):
    X = add_musiq_scores(add_dino_embeddings(amsterdam), variant="ava")
X
```

Create a graph.
Edges are the DINOv2 cosine distance.
Nodes have the aesthetic score as prize.

Find the best path of length k.
An edge goes from i to j when i comes before j in the table.
It exists only when the distance is below the threshold.

```python {.marimo}
G, dist = build_graph(X)
aesthetic = X["aesthetic"].to_numpy()
emb = np.array(X["dino_embeddings"].to_list(), dtype=np.float32)
```

Find the most aesthetic path between similar images.
Cost = DINO distance
Prize = Aesthetic
Budget = the most total cost the path can have

```python {.marimo}
k = mo.ui.slider(start=2, stop=10, step=1, value=5, label="k")
k
```

```python {.marimo}
kk = int(k.value)
flat = np.asarray(dist, dtype=float).ravel()
top_vals = np.partition(flat, -kk)[-kk:]
budget_max = float(top_vals.sum())

coherence_budget = mo.ui.slider(
    start=0, stop=budget_max, step=0.001,
    value=min(np.median(flat), budget_max),
    label="coherence budget",
    debounce=True
)
coherence_budget
```

Find the most aesthetic path within the budget.
The solver is a beam search. It keeps the 50 best partial paths for each last node.

```python {.marimo}
cb = coherence_budget.value
solution = best_budget_path(G, kk, cb)
solution
```

## Score the path

The eval uses the human summaries of the city.
Each human summary is cut to the first k images, so all lists have the same length.
Read the definitions in `evaluation.py`.

```python {.marimo}
mo.stop(solution["path"] is None, mo.md("**No path fits this budget.** Raise the coherence budget or lower k."))
graph_scores = score_systems(
    {"Graph path": X["image_id"][solution["path"]].to_list()},
    city_name,
    all_ids=X["image_id"].to_numpy(),
)
graph_summary = summarise(graph_scores)
graph_summary
```

`normalised` = (system − random) / (human − random).
0 means no better than random. 1 means as good as humans agree with each other.

```python {.marimo}
normalise(graph_summary)
```

```python {.marimo}
mo.ui.altair_chart(comparison_chart(graph_scores))
```

# Visualize the path

```python {.marimo}
path = solution["path"]
mo.stop(path is None, mo.md("**No path fits this budget.** Raise the coherence budget or lower k."))

def show_path_strip(path, X, G, thumb=400):
    n = len(path)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.8))
    axes = np.atleast_1d(axes)
    for pos, (ax, node) in enumerate(zip(axes, path)):
        img = Image.open(X["path"][node]).convert("RGB")
        img.thumbnail((thumb, thumb))
        ax.imshow(img)
        ax.set_title(f"{pos + 1}. img {node}\nprize {G.nodes[node]['prize']:.1f}", fontsize=9)
        ax.axis("off")
    for pos, (u, w) in enumerate(zip(path, path[1:])):
        fig.text((pos + 1) / n, 0.06, f"→ {G[u][w]['weight']:.3f} →",
                 ha="center", fontsize=9, color="dimgray")
    fig.suptitle(f"prize {solution['prize']:.1f}   cost {solution['cost']:.3f} / budget {cb:.3f}")
    fig.subplots_adjust(bottom=0.15, top=0.78, wspace=0.05)
    return fig

show_path_strip(path, X, G)
```

```python {.marimo}
mo.stop(path is None)

xy = TSNE(n_components=2).fit_transform(emb)

fig2, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(xy[:, 0], xy[:, 1], c=aesthetic, cmap="viridis", s=25, alpha=0.6)
fig2.colorbar(sc, ax=ax, label="Prize")

p = xy[path]
for a, b in zip(p, p[1:]):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8))
ax.scatter(p[:, 0], p[:, 1], s=120, facecolors="none", edgecolors="crimson", lw=2)
for pos, (x, y) in enumerate(p):
    ax.text(x, y, f" {pos + 1}", color="crimson", fontsize=10, weight="bold")

ax.set_title("Selected path in DINOv2 space (t-SNE)")
ax.set_xticks([]); ax.set_yticks([])
fig2
```

All sliders in one place

```python {.marimo}
mo.hstack([k, coherence_budget], justify="start", gap=2)
```
