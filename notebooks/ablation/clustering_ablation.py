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
    # Data
    We use the Amsterdam album: 100 images and the human summaries.
    We embed each image with DINOv2.
    """)
    return


@app.cell
def _(load_yfcc_city):
    city = load_yfcc_city("Amsterdam")
    return (city,)


@app.cell
def _(city, np, pl):
    from transformers import pipeline

    encoder = pipeline(
        task="image-feature-extraction",
        model="facebook/dinov2-small-imagenet1k-1-layer",
        pool=True,
        device=0,
    )
    _vecs = [e.squeeze(0).cpu().numpy() for e in encoder(city["path"].to_list(), batch_size=10, return_tensors=True)]
    city_emb = city.with_columns(pl.Series("embedding", np.stack(_vecs)))
    return (city_emb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `X` has one row for each image. scikit-learn cannot use a column of arrays.
    Thus, we divide the embedding into the columns `emb_0` to `emb_383`.
    The metadata columns are `timestamp`, `latitude`, and `longitude`.
    """)
    return


@app.cell
def _(city_emb, pl):
    embeddings = [f"emb_{i}" for i in range(city_emb["embedding"].dtype.size)]
    metadata = ["timestamp", "latitude", "longitude"]

    X = city_emb.select(
        pl.col("embedding").arr.to_struct(embeddings).struct.unnest(),
        pl.col("datetaken").str.to_datetime().dt.epoch("s").cast(pl.Float64).alias("timestamp"),
        "latitude",
        "longitude",
    )
    X
    return X, embeddings, metadata


@app.cell
def _(city_emb, np):
    _ranks = city_emb["human_ranks"].to_list()
    human_picks = [
        np.array([row for row, r in enumerate(_ranks) if r[h] is not None])
        for h in range(len(_ranks[0]))
    ]
    len(human_picks)
    return (human_picks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Shuffled baseline
    Our process takes one image from each cluster.
    Thus, we must find if the clusters divide the album the same way as the humans.

    For each human, the **hit rate** is the number of different clusters that the 10 picks fall in, divided by 10.
    A hit rate of 1 shows that the human picked one image from each cluster.

    Large clusters get more picks by chance.
    Thus, the **shuffled** baseline gives the images to random clusters, but it keeps the cluster sizes of the method.
    We calculate the mean of 1000 shuffles.

    **adjusted** is the hit rate minus the shuffled hit rate. A value of 0 is the same as chance.
    """)
    return


@app.cell
def _(human_picks, np, pl):
    def mean_hitrate(labels: np.ndarray) -> float:
        return float(np.mean([len(np.unique(labels[p])) / len(p) for p in human_picks]))


    def shuffled_hitrate(labels: np.ndarray, n_runs: int = 1000) -> float:
        rng = np.random.default_rng(0)
        return float(np.mean([mean_hitrate(rng.permutation(labels)) for _ in range(n_runs)]))


    def scores(labels: np.ndarray) -> dict:
        real, shuffled = mean_hitrate(labels), shuffled_hitrate(labels)
        return {"hitrate": real, "shuffled": shuffled, "corrected": real - shuffled}


    def score_table(labels: dict[str, np.ndarray]) -> pl.DataFrame:
        return pl.DataFrame([{"method": name} | scores(l) for name, l in labels.items()])

    return (score_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Methods
    We compare three clustering methods: k-means, agglomerative clustering, and a Gaussian mixture (GMM).
    Each method makes 10 clusters. Each human summary has 10 images.
    """)
    return


@app.cell
def _():
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    K = 10
    return (
        AgglomerativeClustering,
        ColumnTransformer,
        GaussianMixture,
        K,
        KMeans,
        Normalizer,
        PCA,
        StandardScaler,
        make_pipeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Visual features, metadata, and PCA
    - The `visual` branch normalizes the embedding, then PCA decreases it to 10 dimensions.
    - The `meta` branch uses the time and the GPS position.
    - `StandardScaler` gives each of the 13 columns the same variance. Thus, the visual part and the metadata part have an effect that is proportional to their number of columns.
    """)
    return


@app.cell
def _(
    AgglomerativeClustering,
    ColumnTransformer,
    GaussianMixture,
    K,
    KMeans,
    Normalizer,
    PCA,
    StandardScaler,
    X,
    embeddings,
    make_pipeline,
    metadata,
    score_table,
):
    _features = ColumnTransformer([
        ("visual", make_pipeline(Normalizer(), PCA(n_components=10, random_state=0), StandardScaler()), embeddings),
        ("meta", StandardScaler(), metadata),
    ])
    meta_pca_labels = {
        "meta+pca | kmeans": make_pipeline(_features, KMeans(n_clusters=K, random_state=0)).fit_predict(X),
        "meta+pca | agglomerative": make_pipeline(_features, AgglomerativeClustering(n_clusters=K)).fit_predict(X),
        "meta+pca | gmm": make_pipeline(_features, GaussianMixture(n_components=K, random_state=0)).fit_predict(X),
    }
    meta_pca_table = score_table(meta_pca_labels)
    meta_pca_table
    return meta_pca_labels, meta_pca_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Visual features only
    The `visual` branch normalizes the embedding. There is no PCA and no metadata.
    """)
    return


@app.cell
def _(
    AgglomerativeClustering,
    ColumnTransformer,
    GaussianMixture,
    K,
    KMeans,
    Normalizer,
    X,
    embeddings,
    make_pipeline,
    score_table,
):
    _features = ColumnTransformer([("visual", Normalizer(), embeddings)])
    visual_labels = {
        "visual | kmeans": make_pipeline(_features, KMeans(n_clusters=K, random_state=0)).fit_predict(X),
        "visual | agglomerative": make_pipeline(_features, AgglomerativeClustering(n_clusters=K)).fit_predict(X),
        "visual | gmm": make_pipeline(_features, GaussianMixture(n_components=K, random_state=0)).fit_predict(X),
    }
    visual_table = score_table(visual_labels)
    visual_table
    return visual_labels, visual_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    All the methods, sorted by the corrected hit rate.
    A negative value shows that the clusters spread the human picks less than random clusters.
    """)
    return


@app.cell
def _(meta_pca_table, pl, visual_table):
    pl.concat([meta_pca_table, visual_table]).sort("corrected", descending=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspect the clusters
    Select a method and a cluster. The table shows the human votes for each cluster.
    """)
    return


@app.cell
def _(meta_pca_labels, mo, visual_labels):
    method_picker = mo.ui.dropdown(
        options=visual_labels | meta_pca_labels, value="visual | kmeans", label="method"
    )
    method_picker
    return (method_picker,)


@app.cell
def _(city_emb, method_picker, pl):
    (
        city_emb.with_columns(cluster=pl.Series(method_picker.value))
        .group_by("cluster")
        .agg(pl.len().alias("size"), pl.col("human_votes").sum().alias("votes"))
        .with_columns((pl.col("votes") / pl.col("size")).alias("votes_per_image"))
        .sort("votes_per_image", descending=True)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Humans pick mostly from the same clusters - not from different ones....

    - Clusters 8 and 3 have 29% of the images, but they get 65% of the votes (130 of 200).
    - Clusters 5 and 6 have 18 images. They get no votes.
    - The mean is 2 votes for each image (200 votes, 100 images).

    the rule "pick one image from each cluster" does not agree with the humans for this album...................
    .
    """)
    return


@app.cell
def _(method_picker, mo):
    cluster_picker = mo.ui.dropdown(
        options=[str(c) for c in sorted(set(method_picker.value))],
        value="0",
        label="cluster",
    )
    cluster_picker
    return (cluster_picker,)


@app.cell
def _(mo, np, pl):
    def cluster_gallery(df: pl.DataFrame, labels: np.ndarray, cluster: int):
        members = (
            df.with_columns(cluster=pl.Series(labels))
            .filter(pl.col("cluster") == cluster)
            .sort("human_votes", descending=True)
        )

        def tile(r):
            return mo.vstack(
                [mo.image(src=r["path"], width=160, height=160),
                 mo.md(f"votes: **{r['human_votes']}**")],
                align="center",
            )

        half = (len(members) + 1) // 2
        rows = [
            mo.hstack([tile(r) for r in part.iter_rows(named=True)], gap=1, justify="start")
            for part in (members.head(half), members.slice(half))
        ]

        return mo.vstack([
            mo.md(f"### Cluster {cluster}: {len(members)} images, {members['human_votes'].sum()} votes"),
            *rows,
        ])

    return (cluster_gallery,)


@app.cell
def _(city_emb, cluster_gallery, cluster_picker, method_picker):
    cluster_gallery(city_emb, method_picker.value, int(cluster_picker.value))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
