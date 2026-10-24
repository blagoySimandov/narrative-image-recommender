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
    FunctionTransformer(to_timestamp),
    MinMaxScaler(),
)
preprocess = make_column_transformer(
    (date_pipeline, ["datetaken"]),
    (FunctionTransformer(to_radians), ["latitude", "longitude"]),
    ("passthrough", ["dino_embeddings"]),
    remainder="drop",
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
from typing import Callable

import kmedoids
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted


class PyKMedoids(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters, metric: str | Callable ="euclidean", random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric  # any sklearn metric name or callable f(a, b) -> float
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
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances, cosine_similarity

K = 10

def distancemetric(v1, v2):
    date1, lat1, lon1, embeddingarray1 = v1[0], v1[1], v1[2], v1[3]
    date2, lat2, lon2, embeddingarray2 = v2[0], v2[1], v2[2], v2[3]
    date_euc = euclidean_distances(date1, date2)
    X,Y = [lat1, lon1], [lat2, lon2]
    latlon_harv = haversine_distances(X, Y)
    cossim = cosine_similarity(embeddingarray1, embeddingarray2)
    return date_euc + latlon_harv + cossim  # add all the distances ??

skpipes = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "clustering",
            PyKMedoids(n_clusters=K, metric=distancemetric),
        ),
    ]
)
```

```python {.marimo}
a = skpipes.fit_predict(df_embedded)
```

```python {.marimo}

```

