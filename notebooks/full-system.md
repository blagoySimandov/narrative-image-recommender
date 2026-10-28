---
title: Full System
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import numpy as np
import polars as pl

from dataset_loaders import load_yfcc_city, load_yfcc_human_summaries
```

# Load dataset

```python {.marimo}
city = load_yfcc_city("amsterdam")
city
```

# Pipeline

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

# Custom KMeans Class using kmedoids

Kmedoids support custom distance function so we are using it.
It also makes a bit more sense for our case since it gives us the closest image to the centroid for "free".

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

## Creating the custom KMedoids distance

```python {.marimo}
from sklearn.metrics.pairwise import cosine_distances, haversine_distances
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
```

```python {.marimo}
from sklearn.pipeline import Pipeline

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
df_clean = df_embedded.with_row_index("orig_idx").filter(
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

## Setting the Weights for Medoid Similarity and Aesthetic Score

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

```python {.marimo}
def order_chronologically(picked: pl.DataFrame) -> pl.DataFrame:
    taken = pl.col("datetaken").str.to_datetime(strict=False)

    return (
        picked.with_columns(taken.alias("taken_at"))
        .sort("taken_at")
        .with_row_index("story_pos")
    )

story = order_chronologically(strategy.value)
story.select("story_pos", "orig_idx", "taken_at", "cluster", "score", "path")
```

```python {.marimo}
def story_card(step: int, row: dict):
    caption = f"**{step}.** {row['taken_at']:%Y-%m-%d %H:%M} · cluster {row['cluster']}"
    return mo.vstack([mo.image(src=row["path"], width=200, height=200), mo.md(caption)], align="center")

mo.hstack( [story_card(i, row) for i, row in enumerate(story.iter_rows(named=True), start=1)], wrap=True, gap=1, justify="start",)
```

## Evaluate against human ground truth using the human_ranks and human_votes columns

Human votes can be used to evaluate whether people truely pick only one image from each cluster.
This can be done via a simple "covered clusters" metric for each human ranking.
So if there are 10 covered clusters from 10 picked images for a certain algorithm such as k means than the accuracy would be 100%
And if there are only 2 covered clusters from 10 picked images than the accuracy would be 20%

A metric for order eval can be used such kendall tau.

The problem is that kendall tau only works when the elements from the two lists are from the same set and
there are no "unpicked" elements from that set - that is it only considers the order without the picking part.

For this evalution to make sense we need to define how we will hanlde the "unpicked" elements.

A sensible definition was given by Rogier van der Geer in
https://xebia.com/blog/using-kendalls-tau-to-compare-recommendations/
The code below is a modifed version of the code from that blog post to work with polars.

```python {.marimo}
import scipy.stats as stats


def extended_tau(list_a, list_b):
    """Calculate the extended Kendall tau from two top-k lists of equal length."""
    n = len(list_a)
    if len(list_b) != n:
        raise ValueError("Both lists must have the same length")

    ranks = join_ranks(create_rank(list_a), create_rank(list_b)).with_columns(
        pl.col("rank_a", "rank_b").fill_null(n)
    )

    # Pad to 2n rows with items that are absent from both lists (tied at rank n)
    n_pad = 2 * n - ranks.height
    dummy_df = pl.DataFrame(
        {"rank_a": [n] * n_pad, "rank_b": [n] * n_pad},
        schema={"rank_a": pl.Int64, "rank_b": pl.Int64},
    )
    total_df = pl.concat([ranks.select("rank_a", "rank_b"), dummy_df])

    tau = stats.kendalltau(total_df["rank_a"].to_numpy(), total_df["rank_b"].to_numpy()).statistic
    return scale_tau(n, tau)


def scale_tau(length, value):
    """Scale an extended tau correlation such that it falls in [-1, +1]."""
    n_0 = 2 * length * (2 * length - 1)
    n_a = length * (length - 1)
    n_d = n_0 - n_a
    min_tau = (2.0 * n_a - n_0) / n_d
    return 2 * (value - min_tau) / (1 - min_tau) - 1


def create_rank(a):
    """Convert an ordered list to a DataFrame of (key, rank)."""
    return pl.DataFrame(
        {"key": a, "rank": pl.int_range(len(a), eager=True, dtype=pl.Int64)}
    )


def join_ranks(rank_a, rank_b):
    """Full outer join of two rank DataFrames on key."""
    return rank_a.rename({"rank": "rank_a"}).join(
        rank_b.rename({"rank": "rank_b"}),
        on="key",
        how="full",
        coalesce=True,
    )
```

## Evaluating the order of the selected images

We use the extended Kendall tau metric to compare the system's ordering against each human ordering, and add human–human and random baselines for context.

```python {.marimo}
import itertools

def tau_table(pairs, avg_label, **cols):
    """Per-pair tau scores plus an average row."""
    per = pl.DataFrame({"pair": pairs, **{c: np.asarray(v, dtype=float) for c, v in cols.items()}})
    avg = pl.DataFrame({"pair": [avg_label], **{c: [float(np.mean(v))] for c, v in cols.items()}})
    return pl.concat([per, avg])
```

```python {.marimo}
humanranks = [np.asarray(r) for r in load_yfcc_human_summaries("amsterdam")]
hn = len(humanranks)
pred = story["image_id"].to_numpy()
all_ids = df_clean["image_id"].to_numpy()
```

Check that every image the humans picked is in `df_clean`, so the system had a fair chance of selecting it.

```python {.marimo}
_missing = set().union(*map(set, humanranks)) - set(all_ids)
assert not _missing, f"{len(_missing)} human-picked images missing from df_clean"
```

### System vs Human

```python {.marimo}
scores_s2h = [float(extended_tau(pred, h)) for h in humanranks]
results_s2h = tau_table(
    [f"system_vs_human_{i}" for i in range(hn)],
    "system_vs_humans_avg",
    tau=scores_s2h,
)
results_s2h
```

### Human vs Human

Do humans agree with each other on the order?

```python {.marimo}
_pairs = list(itertools.combinations(range(hn), 2))
scores_h2h = [float(extended_tau(humanranks[i], humanranks[j])) for i, j in _pairs]
results_h2h = tau_table(
    [f"human_vs_human_{i}_{j}" for i, j in _pairs],
    "human_vs_humans_avg",
    tau=scores_h2h,
)
results_h2h
```

### Random baseline vs Human

Random selection picks k random images from `df_clean` in random order.

```python {.marimo}
_rng = np.random.default_rng(0)  # fixed seed for reproducibility
n_trials = 200
k = len(pred)

rand_sel = np.empty((n_trials, hn))

for _t in range(n_trials):
    _sample = _rng.choice(all_ids, size=k, replace=False)
    for _i, _h in enumerate(humanranks):
        rand_sel[_t, _i] = extended_tau(_sample, _h)

results_r2h = tau_table(
    [f"random_vs_human_{i}" for i in range(hn)],
    "random_vs_humans_avg",
    tau=rand_sel.mean(axis=0),
)
results_r2h
```

### Summary

```python {.marimo}
pl.concat([
    results_s2h.tail(1),
    results_h2h.tail(1),
    results_r2h.tail(1),
])
```

## Selection overlap

Overlap is the fraction of images two lists share: shared / list length, using the longer list if lengths differ. 1.0 means the same images were picked, 0.0 means none in common.

```python {.marimo}
def overlap(a, b):
    return len(set(a) & set(b)) / max(len(a), len(b))
```

### System vs Human

```python {.marimo}
ov_s2h = tau_table(
    [f"system_vs_human_{i}" for i in range(hn)],
    "system_vs_humans_avg",
    overlap=[overlap(pred, h) for h in humanranks],
)
ov_s2h
```

### Human vs Human

```python {.marimo}
_pairs = list(itertools.combinations(range(hn), 2))
ov_h2h = tau_table(
    [f"human_vs_human_{i}_{j}" for i, j in _pairs],
    "human_vs_humans_avg",
    overlap=[overlap(humanranks[i], humanranks[j]) for i, j in _pairs],
)
ov_h2h
```

### Random vs Human

```python {.marimo}
_rng = np.random.default_rng(0)
_ov = np.empty((200, hn))
for _t in range(200):
    _sample = _rng.choice(all_ids, size=len(pred), replace=False)
    _ov[_t] = [overlap(_sample, h) for h in humanranks]

ov_r2h = tau_table(
    [f"random_vs_human_{i}" for i in range(hn)],
    "random_vs_humans_avg",
    overlap=_ov.mean(axis=0),
)
ov_r2h
```

### Overlap Summary

```python {.marimo}
pl.concat([ov_s2h.tail(1), ov_h2h.tail(1), ov_r2h.tail(1)])
```

## F-Measure evaluation

The F-measure compares the set of selected images with the set of reference images.
Precision is the shared images divided by the selected images.
Recall is the shared images divided by the reference images.
F = 2 × shared / (selected + reference).
If the two lists have the same length, F is equal to the overlap.

```python {.marimo}
def f_measure(pred, ref):
    pred_set, ref_set = set(pred), set(ref)
    return 2 * len(pred_set & ref_set) / (len(pred_set) + len(ref_set))
```

### System vs Human

```python {.marimo}
fm_s2h = tau_table(
    [f"system_vs_human_{i}" for i in range(hn)],
    "system_vs_humans_avg",
    f=[f_measure(pred, h) for h in humanranks],
)
fm_s2h
```

### Human vs Human

```python {.marimo}
_pairs = list(itertools.combinations(range(hn), 2))
fm_h2h = tau_table(
    [f"human_vs_human_{i}_{j}" for i, j in _pairs],
    "human_vs_humans_avg",
    f=[f_measure(humanranks[i], humanranks[j]) for i, j in _pairs],
)
fm_h2h
```

### Random vs Human

```python {.marimo}
_rng = np.random.default_rng(0)
_fm = np.empty((200, hn))
for _t in range(200):
    _sample = _rng.choice(all_ids, size=len(pred), replace=False)
    _fm[_t] = [f_measure(_sample, h) for h in humanranks]

fm_r2h = tau_table(
    [f"random_vs_human_{i}" for i in range(hn)],
    "random_vs_humans_avg",
    f=_fm.mean(axis=0),
)
fm_r2h
```

### F-Measure Summary

```python {.marimo}
pl.concat([fm_s2h.tail(1), fm_h2h.tail(1), fm_r2h.tail(1)])
```

## Comparison: random vs system vs human

Each point is one pair (random rows are already averaged over 200 trials per human).
Random is the floor, human–human agreement is the ceiling, and the system should sit between.

```python {.marimo}
import altair as alt  # drop this line if altair is already imported in another cell
```

```python {.marimo}
def _long(df, comparison, col, metric):
    """Per-pair rows only (drops the *_avg row), in long format."""
    return (
        df.filter(~pl.col("pair").str.ends_with("_avg"))
        .select(
            pl.lit(comparison).alias("comparison"),
            pl.lit(metric).alias("metric"),
            pl.col(col).cast(pl.Float64).alias("value"),
        )
    )

all_scores = pl.concat([
    _long(results_r2h, "Random vs Human", "tau", "Kendall τ (order)"),
    _long(results_s2h, "System vs Human", "tau", "Kendall τ (order)"),
    _long(results_h2h, "Human vs Human", "tau", "Kendall τ (order)"),
    _long(ov_r2h, "Random vs Human", "overlap", "Overlap"),
    _long(ov_s2h, "System vs Human", "overlap", "Overlap"),
    _long(ov_h2h, "Human vs Human", "overlap", "Overlap"),
    _long(fm_r2h, "Random vs Human", "f", "F-measure"),
    _long(fm_s2h, "System vs Human", "f", "F-measure"),
    _long(fm_h2h, "Human vs Human", "f", "F-measure"),
])
COMPARISON_ORDER = ["Random vs Human", "System vs Human", "Human vs Human"]
```

### Summary table

`normalised` = (system − random) / (human − random): 0 means no better than random,
1 means as good as humans agree with each other.

```python {.marimo}
_means = (
    all_scores.group_by("metric", "comparison")
    .agg(pl.col("value").mean().alias("mean"))
    .pivot(on="comparison", index="metric", values="mean")
)
comparison_summary = _means.select(
    "metric",
    pl.col("Random vs Human").alias("random"),
    pl.col("System vs Human").alias("system"),
    pl.col("Human vs Human").alias("human"),
).with_columns(
    ((pl.col("system") - pl.col("random")) / (pl.col("human") - pl.col("random")))
    .alias("normalised")
)
comparison_summary
```

### Chart - ai generated chart

```python {.marimo}
_base = alt.Chart(all_scores).encode(
    x=alt.X("comparison:N", sort=COMPARISON_ORDER, title=None,
            axis=alt.Axis(labelAngle=-20)),
    color=alt.Color("comparison:N", sort=COMPARISON_ORDER, legend=None,
                    scale=alt.Scale(range=["#9aa0a6", "#c2387a", "#1d64d8"])),
)
_points = _base.mark_circle(size=55, opacity=0.55).encode(
    y=alt.Y("value:Q", title="score"),
    xOffset=alt.XOffset("jitter:Q"),
    tooltip=["comparison", "metric", alt.Tooltip("value:Q", format=".3f")],
).transform_calculate(jitter="random()")
_mean = _base.mark_tick(thickness=3, size=34).encode(y="mean(value):Q")
_err = _base.mark_errorbar(extent="stdev").encode(y=alt.Y("value:Q", title="score"))

comparison_chart = (
    alt.layer(_points, _err, _mean, data=all_scores)
    .properties(width=220, height=260)
    .facet(column=alt.Column("metric:N", title=None))
    .resolve_scale(y="independent")
)
mo.ui.altair_chart(comparison_chart)
```