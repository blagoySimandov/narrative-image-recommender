import kmedoids
import numpy as np
import polars as pl
import sklearn
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.compose import make_column_transformer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    haversine_distances,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.utils.validation import check_is_fitted


class PyKMedoids(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters, metric: str = "euclidean", random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        D = pairwise_distances(X, metric=self.metric)
        result = kmedoids.fasterpam(D, self.n_clusters, random_state=self.random_state)
        self.labels_ = np.asarray(result.labels)
        self.medoid_indices_ = np.asarray(result.medoids)
        self.cluster_centers_ = X[self.medoid_indices_]
        self.inertia_ = result.loss
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        X = np.asarray(X, dtype=np.float64)
        return pairwise_distances(X, self.cluster_centers_, metric=self.metric).argmin(axis=1)


def to_radians(X):
    return np.radians(np.asarray(X, dtype=np.float64))


def to_timestamp(X):
    return (
        pl.Series(np.ravel(X))
        .str.to_datetime(strict=False)
        .dt.epoch("s")
        .to_numpy()
        .reshape(-1, 1)
    )


def _scaled(D):
    return D / D.max()


def combined_distance(X, weights=(1.0, 1.0, 1.0)):
    t = X["datetaken"].to_numpy().reshape(-1, 1)
    geo = X.select("latitude", "longitude").to_numpy()
    emb = X["dino_embeddings"].to_numpy()
    parts = [euclidean_distances(t - t.T), haversine_distances(geo), cosine_distances(emb)]
    return sum(w * _scaled(D) for w, D in zip(weights, parts))


def make_cluster_pipeline(n_clusters: int) -> Pipeline:
    date_pipeline = make_pipeline(
        FunctionTransformer(to_timestamp, feature_names_out="one-to-one"),
        MinMaxScaler(),
    )
    preprocess = make_column_transformer(
        (date_pipeline, ["datetaken"]),
        (FunctionTransformer(to_radians, feature_names_out="one-to-one"), ["latitude", "longitude"]),
        ("passthrough", ["dino_embeddings"]),
        remainder="drop",
        verbose_feature_names_out=False,
    )
    distance = FunctionTransformer(combined_distance).set_output(transform="default")
    clustering = PyKMedoids(n_clusters=n_clusters, metric="precomputed", random_state=0)
    return Pipeline([("preprocess", preprocess), ("distance", distance), ("clustering", clustering)])


def medoid_similarity(clusterer, labels) -> np.ndarray:
    return 1 - clusterer.cluster_centers_[labels, np.arange(len(labels))]


def _minmax(col: str) -> pl.Expr:
    return (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())


def add_selection_score(df: pl.DataFrame, w_sim: float, w_aes: float) -> pl.DataFrame:
    return df.with_columns(
        (w_sim * _minmax("medoid_sim") + w_aes * _minmax("aesthetic")).alias("score")
    )


def cluster_photos(df: pl.DataFrame, n_clusters: int) -> pl.DataFrame:
    clean = df.with_row_index("orig_idx").filter(
        pl.col("datetaken").str.to_datetime(strict=False).is_not_null()
    )
    pipe = make_cluster_pipeline(n_clusters)
    with sklearn.config_context(transform_output="polars"):
        labels = pipe.fit_predict(clean)
    sim = medoid_similarity(pipe["clustering"], labels)
    return clean.with_columns(pl.Series("cluster", labels), pl.Series("medoid_sim", sim))


def rank_photos(
    df: pl.DataFrame, n_clusters: int, w_sim: float = 0.5, w_aes: float = 0.5
) -> pl.DataFrame:
    return add_selection_score(cluster_photos(df, n_clusters), w_sim, w_aes)


def top_per_cluster(df: pl.DataFrame, quotas: pl.DataFrame) -> pl.DataFrame:
    rank = pl.col("score").rank("ordinal", descending=True).over("cluster")
    return df.join(quotas, on="cluster").filter(rank <= pl.col("quota")).drop("quota")


def proportional_quotas(df: pl.DataFrame, n: int) -> pl.DataFrame:
    sizes = df["cluster"].value_counts().sort("cluster")
    raw = sizes["count"].to_numpy() / df.height * n
    quota = np.floor(raw).astype(np.int64)
    quota[np.argsort(quota - raw)[: n - quota.sum()]] += 1
    return pl.DataFrame({"cluster": sizes["cluster"], "quota": quota})


def pick_photos(ranked: pl.DataFrame, strategy: str, n: int) -> pl.DataFrame:
    if strategy == "coverage":
        quotas = ranked.select(pl.col("cluster").unique(), quota=pl.lit(1, dtype=pl.Int64))
    else:
        quotas = proportional_quotas(ranked, n)
    return top_per_cluster(ranked, quotas)


def order_chronologically(picked: pl.DataFrame) -> pl.DataFrame:
    taken = pl.col("datetaken").str.to_datetime(strict=False)
    return picked.with_columns(taken.alias("taken_at")).sort("taken_at").with_row_index("story_pos")


def select_full_system(
    df: pl.DataFrame,
    k: int,
    strategy: str = "coverage",
    w_sim: float = 0.5,
    w_aes: float = 0.5,
) -> pl.DataFrame:
    ranked = rank_photos(df, n_clusters=k, w_sim=w_sim, w_aes=w_aes)
    return order_chronologically(pick_photos(ranked, strategy, k))
