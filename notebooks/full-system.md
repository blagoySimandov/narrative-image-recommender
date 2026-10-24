---
title: Full System
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import numpy as np
import polars as pl

from dataset_loaders import load_yfcc_city
```

# Load dataset

```python {.marimo}
city = load_yfcc_city("amsterdam")
city
```

# Pipelinee

```python {.marimo}
from transformers import pipeline

dino_pipe = pipeline(task="image-feature-extraction", model="facebook/dinov2-small")
```

```python {.marimo}
def add_dino_embeddings(
    df: pl.DataFrame, col: str, batch_size: int = 32
) -> pl.DataFrame:
    out = dino_pipe(df[col].to_list(), pool=True, batch_size=batch_size)
    emb = np.asarray(out, dtype=np.float32).reshape(len(df), -1)
    return df.with_columns(pl.Series("dino_embeddings", emb))

df_embedded = city.pipe(add_dino_embeddings, "path")
```

```python {.marimo}
df_embedded.head(n=2)
```

```python {.marimo}
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

set_config(transform_output="polars")

def to_radians(X):
    return np.radians(np.asarray(X, dtype=np.float64))

def to_timestamp(X):
    return (
        pl.Series(np.ravel(X))
        .str.to_datetime(strict=False)  # unparseable → null (like errors="coerce")
        .dt.epoch("s")  # seconds since epoch
        .to_numpy()
        .reshape(-1, 1)
    )

date_pipeline = make_pipeline(
    FunctionTransformer(to_timestamp, feature_names_out="one-to-one"),
    MinMaxScaler(),
)
preprocess = make_column_transformer(
    (date_pipeline, ["datetaken"]),
    (
        FunctionTransformer(to_radians, feature_names_out="one-to-one"),
        ["latitude", "longitude"],
    ),
    ("passthrough", ["dino_embeddings"]),
    remainder="drop",
    verbose_feature_names_out=False,
)
```

## Check how the data looks after preprocessing

```python {.marimo}
preprocess.fit_transform(df_embedded)
```

# Custom KMeans Class uing pyclustering Kmeans

We need to use pyclustering's kmeans since it's the only one that supports custom distance metrics definitions
The other option is the kmedoid library but it supports only the "kmedoids" algorithms.

```python {.marimo}
import kmedoids
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted


class PyKMedoids(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters, metric: str = "euclidean", random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        D = pairwise_distances(X, metric=self.metric)

        result = kmedoids.fasterpam(
            D, self.n_clusters, random_state=self.random_state
        )

        self.labels_ = np.asarray(result.labels)
        self.medoid_indices_ = np.asarray(result.medoids)
        self.cluster_centers_ = X[self.medoid_indices_]
        self.inertia_ = result.loss
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        X = np.asarray(X, dtype=np.float64)
        return pairwise_distances(
            X, self.cluster_centers_, metric=self.metric
        ).argmin(axis=1)
```

```python {.marimo}
from sklearn.metrics.pairwise import cosine_distances, haversine_distances
from sklearn.pipeline import Pipeline

K = 10

def _scaled(D):
    return D / D.max()

def combined_distance(X, weights=(1.0, 1.0, 1.0)):
    t = X["datetaken"].to_numpy().reshape(-1, 1)
    geo = X.select("latitude", "longitude").to_numpy()
    emb = X["dino_embeddings"].to_numpy()
    parts = [
        np.abs(t - t.T),
        haversine_distances(geo),
        cosine_distances(emb),
    ]
    return sum(w * _scaled(D) for w, D in zip(weights, parts))

skpipes = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "distance",
            FunctionTransformer(combined_distance).set_output(transform="default"),
        ),
        (
            "clustering",
            PyKMedoids(n_clusters=K, metric="precomputed", random_state=0),
        ),
    ]
)
```

```python {.marimo}
df_clean = df_embedded.filter(
    pl.col("datetaken").str.to_datetime(strict=False).is_not_null()
)
labels = skpipes.fit_predict(df_clean)
df_clustered = df_clean.with_columns(pl.Series("cluster", labels))
df_clustered
```

## Selecting images from the cluster

From each cluster we will select a combination of the most represntative image and the most aesthetic one.
We will define 2 weights w1 and w2 each of which will be between 0 and 1 and will correspond respecively to the
similarty of the selected image to the center of the cluster and 2 the aestheticity of the image.

Seince we are using Kmedoids and not KMeans the most similar image will clearly be the medoid.

Another selection strategy will try to employ is to pick more images from the cluster that is bigger -
The idea comes after an expirremnt where we saw that poeple don't necessarily pick the most diverse images for their "summarization"
And instead tend to pick images in correaltion to the cluster size - If there are 40 images of boats out of a 100 - there is a good
change people would pick 4/10 of the images from the boat cluster.

This does go against our initial hypothesis about ensuring maximum coverage since now smaller clusters would not be represented.

---

Loading MUSIQ - a model for scoring aesthicity of images

```python {.marimo}
import tensorflow as tf
import tensorflow_hub as hub

musiq = hub.load("https://www.kaggle.com/models/google/musiq/tensorFlow2/koniq-10k/1")
musiq_fn = musiq.signatures["serving_default"]
```

```python {.marimo}
from pathlib import Path

def musiq_score(path: str) -> float:
    img_bytes = tf.constant(Path(path).read_bytes())
    return float(list(musiq_fn(img_bytes).values())[0].numpy())

def medoid_similarity(clusterer, labels) -> np.ndarray:
    D_to_medoids = clusterer.cluster_centers_
    return 1 - D_to_medoids[labels, np.arange(len(labels))]

df_scored = df_clustered.with_columns(
    pl.col("path").map_elements(musiq_score, return_dtype=pl.Float64).alias("aesthetic"),
    pl.Series("medoid_sim", medoid_similarity(skpipes["clustering"], labels)),
)
```

```python {.marimo}
w1 = mo.ui.slider(label="w1 (medoid similarity)", start=0.0, stop=1.0, step=0.05, value=0.5)
w2 = mo.ui.slider(label="w2 (aesthetic)", start=0.0, stop=1.0, step=0.05, value=0.5)
n_select = mo.ui.slider(label="images to select", start=1, stop=K * 3, step=1, value=K)
mo.vstack([w1, w2, n_select])
```

```python {.marimo}
#scaling the medoid similarity to [0, 1] and aesthetic to [0, 1]
def _minmax(col: str) -> pl.Expr:
    return (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())

def add_selection_score(df: pl.DataFrame, w_sim: float, w_aes: float) -> pl.DataFrame:
    return df.with_columns(
        (w_sim * _minmax("medoid_sim") + w_aes * _minmax("aesthetic")).alias("score")
    )

def top_per_cluster(df: pl.DataFrame, quotas: pl.DataFrame) -> pl.DataFrame:
    rank = pl.col("score").rank("ordinal", descending=True).over("cluster")
    return df.join(quotas, on="cluster").filter(rank <= pl.col("quota")).drop("quota")

df_ranked = add_selection_score(df_scored, w1.value, w2.value)
```

```python {.marimo}
#one image per cluster (uniform distribution)
one_quotas = df_ranked.select(pl.col("cluster").unique(), quota=pl.lit(1, dtype=pl.Int64))
picked_coverage = top_per_cluster(df_ranked, one_quotas)
picked_coverage
```

```python {.marimo}
def proportional_quotas(df: pl.DataFrame, n: int) -> pl.DataFrame:
    sizes = df["cluster"].value_counts().sort("cluster")
    raw = sizes["count"].to_numpy() / df.height * n
    quota = np.floor(raw).astype(np.int64)
    leftover = n - quota.sum()
    quota[np.argsort(quota - raw)[:leftover]] += 1
    return pl.DataFrame({"cluster": sizes["cluster"], "quota": quota})

picked_proportional = top_per_cluster(
    df_ranked, proportional_quotas(df_ranked, n_select.value) #type: ignore
)
picked_proportional
```

## Visualize the selected images against the images simliar images

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

For the sake of simplicity we can order the iamges using their datetaken column. We will only use chronological ordering
for now and try other more innovatinve techniques later.

## Evaluate against human ground truth using the human_ranks and human_votes columns

Human votes can be used to evaluate whether people trule pick only one image from each cluster.
This can be done via a simple "covered clusters" metric for each human ranking.
So if there are 10 covered clusters from 10 picked images for a certain algorithm such as k means than the accuracy would be 100%
And if there are only 2 covered clusters from 10 picked images than the accuracy would be 20%

## Evaluating the order of the selected images.

TODO

```


```