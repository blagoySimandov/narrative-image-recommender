---
title: Full System
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import polars as pl

from dataset_loaders import load_yfcc_city
from evaluation import comparison_chart, normalise, score_systems, summarise
from features import add_dino_embeddings, add_musiq_scores
from systems.full_system import (
    add_selection_score,
    cluster_photos,
    order_chronologically,
    pick_photos,
)
```

The code for this system is in `systems/full_system.py`.
The code for the eval is in `evaluation.py`.
Use `notebooks/compare.md` to compare this system with others.

# Load dataset

```python {.marimo}
city_name = "amsterdam"
city = load_yfcc_city(city_name)
city
```

# Features

DINOv2 embeddings and the MUSIQ `koniq-10k` aesthetic score.

```python {.marimo}
with mo.persistent_cache(name=f"features-{city_name}-koniq"):
    df_features = add_musiq_scores(add_dino_embeddings(city), variant="koniq-10k")
df_features.head(n=2)
```

# Cluster

KMedoids on a combined distance: time, GPS and DINOv2 cosine.
KMedoids gives the most representative photo of a cluster, the medoid, for free.

```python {.marimo}
K = 10
df_clustered = cluster_photos(df_features, n_clusters=K)
df_clustered
```

## Selecting images from the cluster

From each cluster we select a combination of the most representative image and the most aesthetic one.
The weights w1 and w2 are between 0 and 1.
w1 is the weight of the similarity to the medoid. w2 is the weight of the aesthetic score.

Another strategy picks more images from the bigger clusters.
In an earlier experiment people did not pick the most diverse images for a summary.
They picked in proportion to the cluster size.
If 40 of 100 images show boats, people pick about 4 of 10 images from the boat cluster.
This goes against our first hypothesis about maximum coverage, because small clusters are not represented.

```python {.marimo}
w1 = mo.ui.slider(label="w1 (medoid similarity)", start=0.0, stop=1.0, step=0.05, value=0.5)
w2 = mo.ui.slider(label="w2 (aesthetic)", start=0.0, stop=1.0, step=0.05, value=0.5)
n_select = mo.ui.slider(label="images to select", start=1, stop=K * 3, step=1, value=K)
mo.vstack([w1, w2, n_select])
```

```python {.marimo}
df_ranked = add_selection_score(df_clustered, w1.value, w2.value)
picked_coverage = pick_photos(df_ranked, "coverage", n_select.value)
picked_proportional = pick_photos(df_ranked, "proportional", n_select.value)
```

## Visualize the selected images against the similar images

```python {.marimo}
strategy = mo.ui.dropdown(
    options={"coverage": picked_coverage, "proportional": picked_proportional},
    value="coverage",
    label="selection strategy",
)
max_members = mo.ui.slider(label="members to show", start=5, stop=50, step=5, value=20)
mo.hstack([strategy, max_members])
```

```python {.marimo}
def _thumbs(paths: list[str], size: int):
    return mo.hstack([mo.image(src=p, width=size, height=size) for p in paths], wrap=True, gap=0.5)

def cluster_view(cluster_id, picked: pl.DataFrame, n_members: int):
    members = (
        df_ranked.filter(pl.col("cluster") == cluster_id)
        .join(picked.select("path"), on="path", how="anti")
        .sort("medoid_sim", descending=True)
        .head(n_members)
    )
    size = df_ranked.filter(pl.col("cluster") == cluster_id).height
    return mo.vstack([
        mo.md(f"### Cluster {cluster_id} ({size} images, {picked.height} picked)"),
        mo.hstack([_thumbs(picked["path"].to_list(), 180), _thumbs(members["path"].to_list(), 90)],
                  gap=2, align="start", widths=[1, 3]),
        mo.md("---"),
    ])

_picked = strategy.value
mo.vstack([
    cluster_view(c, _picked.filter(pl.col("cluster") == c), max_members.value)
    for c in _picked["cluster"].unique().sort()
])
```

## Order images

For simplicity we order the images by their `datetaken` column.
We will try other orderings later.

```python {.marimo}
story = order_chronologically(strategy.value)
story.select("story_pos", "orig_idx", "taken_at", "cluster", "score", "path")
```

```python {.marimo}
def story_card(step: int, row: dict):
    caption = f"**{step}.** {row['taken_at']:%Y-%m-%d %H:%M} · cluster {row['cluster']}"
    return mo.vstack([mo.image(src=row["path"], width=200, height=200), mo.md(caption)], align="center")

mo.hstack([story_card(i, row) for i, row in enumerate(story.iter_rows(named=True), start=1)], wrap=True, gap=1, justify="start")
```

## Evaluate against human ground truth

The eval uses the human summaries of the city.
Each human summary is cut to the first k images, so all lists have the same length.
The metrics are extended Kendall tau (order), overlap and F-measure.
Random is the floor. Human vs human is the ceiling.
Read the definitions in `evaluation.py`.

```python {.marimo}
scores = score_systems(
    {"Full system": story["image_id"].to_list()},
    city_name,
    all_ids=df_features["image_id"].to_numpy(),
)
summary = summarise(scores)
summary
```

`normalised` = (system − random) / (human − random).
0 means no better than random. 1 means as good as humans agree with each other.

```python {.marimo}
normalise(summary)
```

```python {.marimo}
mo.ui.altair_chart(comparison_chart(scores))
```
