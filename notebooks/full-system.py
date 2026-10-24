import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from dataset_loaders import load_yfcc_city

    return load_yfcc_city, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load dataset
    """)
    return


@app.cell
def _(load_yfcc_city):
    city = load_yfcc_city("amsterdam")
    city
    return (city,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipelinee
    """)
    return


@app.cell
def _():
    from transformers import pipeline

    dino_pipe = pipeline(task="image-feature-extraction", model="facebook/dinov2-small")
    return (dino_pipe,)


@app.cell
def _(city, dino_pipe, np, pl):
    def add_dino_embeddings(
        df: pl.DataFrame, col: str, batch_size: int = 32
    ) -> pl.DataFrame:
        out = dino_pipe(df[col].to_list(), pool=True, batch_size=batch_size)
        emb = np.asarray(out, dtype=np.float32).reshape(len(df), -1)
        return df.with_columns(pl.Series("dino_embeddings", emb))

    df_embedded = city.pipe(add_dino_embeddings, "path")
    return (df_embedded,)


@app.cell
def _(df_embedded):
    df_embedded.head(n=2)
    return


@app.cell
def _(np, pl):
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
    return (preprocess,)


@app.cell
def _(df_embedded, preprocess):
    preprocess.fit_transform(df_embedded)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Custom KMeans Class uing pyclustering Kmeans
    We need to use pyclustering's kmeans since it's the only one that supports custom distance metrics definitions
    The other option is the kmedoid library but it supports only the "kmedoids" algorithms.
    """)
    return


@app.cell
def _(np):
    import kmedoids
    from sklearn.base import BaseEstimator, ClusterMixin
    from sklearn.metrics import pairwise_distances
    from sklearn.utils.validation import check_is_fitted

    class PyKMedoids(ClusterMixin, BaseEstimator):
        def __init__(self, n_clusters, metric="euclidean", random_state=None):
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

    return (PyKMedoids,)


@app.cell
def _(
    PyKMedoids,
    cosine_similarity,
    euclidean_distances,
    haversine_distances,
    preprocess,
    v,
):
    from sklearn.pipeline import Pipeline

    K = 10

    def distancemetric(v1, v2):
        date1, lat1, lon1, embeddingarray1 = v1[0], v1[1], v1[2], v[3]
        date2, lat2, lon2, embeddingarray2 = v2[0], v2[1], v2[2], v[3]
        date_euc = euclidean_distances(date1, date2)
        latlon_harv = haversine_distances(lat1, lon1, lat2, lon2)
        embeddings = cosine_similarity(embeddingarray1, embeddingarray2)
        return date_euc + latlon_harv + embeddings  # add all the distances ??

    skpipes = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "clustering",
                PyKMedoids(n_clusters=K),
            ),
        ]
    )
    return (skpipes,)


@app.cell
def _(df_embedded, skpipes):
    a = skpipes.fit_predict(df_embedded)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
