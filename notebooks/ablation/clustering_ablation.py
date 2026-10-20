import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from dataset_loaders import load_yfcc_city

    return load_yfcc_city, mo, pl


@app.cell
def _(load_yfcc_city):
    city = load_yfcc_city("amsterdam")
    city
    return (city,)


@app.cell
def _():
    from transformers import pipeline

    pipe = pipeline(
        task="image-feature-extraction",
        model="facebook/dinov2-small-imagenet1k-1-layer",
        pool=True,
        device=0,
    )
    return (pipe,)


@app.cell
def _(city, pipe, pl):
    import numpy as np
    from tqdm.auto import tqdm
    from sklearn.preprocessing import normalize

    paths = city["path"].to_list()

    vecs = []
    for emb in tqdm(
        pipe(paths, batch_size=10, return_tensors=True), total=len(paths)
    ):
        vecs.append(emb.squeeze(0).detach().cpu().numpy())

    city_emb = city.with_columns(
        pl.Series(
            "embedding", normalize(np.stack(vecs))
        )  # normalize the embedding
    )
    return city_emb, np, paths


@app.cell
def _(city_emb):
    city_emb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run clustering algorithms and compare agains the human ground truth
    1. K means
    2. Agglomerative clustering
    3. Gaussian
    4. Random k
    """)
    return


@app.cell
def _():
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture

    random_state = 0
    K = 10  # stickign to 10 images since that's what the human ground truth is
    return AgglomerativeClustering, GaussianMixture, K, KMeans, random_state


@app.cell
def _(
    AgglomerativeClustering,
    GaussianMixture,
    K,
    KMeans,
    city_emb,
    random_state,
):
    kmeans = KMeans(n_clusters=K, random_state=random_state).fit(
        city_emb["embedding"]
    )
    aglomerative = AgglomerativeClustering(n_clusters=K).fit(
        city_emb["embedding"]
    )
    gaussian = GaussianMixture(n_components=K, random_state=random_state).fit(
        city_emb["embedding"]
    )
    return aglomerative, gaussian, kmeans


@app.cell
def _(kmeans):
    kmeans_labels = kmeans.labels_
    kmeans_labels
    return (kmeans_labels,)


@app.cell
def _(aglomerative):
    a_labels = aglomerative.labels_
    a_labels
    return (a_labels,)


@app.cell
def _(city_emb, gaussian):
    g_labels = gaussian.predict(city_emb["embedding"])
    g_labels
    return (g_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualize these.
    Since they are N dimensionnal use PCA to reduce dimensiolanliyty to 2d and show that way in the fig
    """)
    return


@app.cell
def _(a_labels, city_emb, g_labels, kmeans_labels, np):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    X = np.vstack(city_emb["embedding"])  # (n_cities, dim)
    X2 = PCA(n_components=2).fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, labels, title in zip(
        axes,
        [kmeans_labels, a_labels, g_labels],
        ["K-Means", "Agglomerative", "GMM"],
    ):
        ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=40)
        score = silhouette_score(X, labels, metric="cosine")
        ax.set_title(title)
        ax.text(
            0.5,
            -0.1,
            f"Silhouette: {score:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
        )

    fig.text(
        0.5,
        0.01,
        "PCA projection of city embeddings",
        ha="center",
        fontsize=12,
    )
    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )  # leave room at the bottom for the caption
    plt.show()
    return PCA, X, X2, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian labels get stuck and equal  the k means ones
    GMM starts off with K mean


    Why ?
    We have more dimensions that we have clusters....
    """)
    return


@app.cell
def _(g_labels, kmeans_labels):
    kmeans_labels == g_labels
    return


@app.cell
def _(X2, a_labels, np, paths, plt):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image, ImageDraw

    _fig, _ax = plt.subplots(figsize=(14, 10), dpi=200)
    for (x, y), p, c in zip(X2, paths, a_labels):
        im = Image.open(p).convert("RGB")
        im.thumbnail((160, 160))
        ab = AnnotationBbox(
            OffsetImage(np.asarray(im), zoom=0.15),
            (x, y),
            frameon=True,
            bboxprops=dict(edgecolor=plt.cm.tab10(c % 10), lw=2),
        )
        _ax.add_artist(ab)

    _ax.set_xlim(X2[:, 0].min() - 0.05, X2[:, 0].max() + 0.05)
    _ax.set_ylim(X2[:, 1].min() - 0.05, X2[:, 1].max() + 0.05)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evalute against human ground truth
    All models against human GD!


    The `city_emb` dataframe has `human_ranks` and `human_votes`
    human ranks is an array showing at which position the ith human put the image or null if he didn't
    human votes is just how many humans picked this image in the summary at all.


    ----
    We will compare each human collection with the most similar image to the centroid.
    """)
    return


@app.cell
def _(city_emb):
    city_emb
    return


@app.cell
def _(city_emb, pl):
    def get_human_collections(city_emb):
        n_humans = city_emb["human_ranks"].list.len().max()
        return [
            city_emb.filter(
                pl.col("human_ranks")
                .list.get(i, null_on_oob=True)
                .is_not_null()
            )
            for i in range(n_humans)
        ]

    human_collections = get_human_collections(city_emb)
    human_collections[1]
    return (human_collections,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Process
    Our process assumes we take 1 image from each cluster. For this study do make sense the thing we should check
    is whether we correctly divide the images into clusters. and from that point on we can see how the human images map over to the clusters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #  Do the clusters agree with the humans?

    The question here is whether each annotator picks about one image per cluster. If they do, our clusters match the way people split the album into "moments".


    For each human's 10 picks, count:
    - Clusters hit: the number of different clusters the 10 picks fall in (best is 10).
    - Collisions: the number of picks that share a cluster with another pick.
    """)
    return


@app.cell
def _(kmeans_labels, np):
    rng = np.random.default_rng(0)
    dummy_labels = [
        rng.permutation(kmeans_labels) for _ in range(1000)
    ]  # "dummy" clusterer is our baseline.  shuffled kmeans - keep the cluster sizes the same for the sake of the xperiment
    return (dummy_labels,)


@app.cell
def _(dummy_labels):
    dummy_labels
    return


@app.cell
def _(K, np, paths, pl):
    path_to_idx = {p: i for i, p in enumerate(paths)}


    def score_one_human(human_collection: pl.DataFrame, cluster_labels: np.ndarray) -> dict:
        picks = np.array([path_to_idx[p] for p in human_collection["path"]])
        clusters_hit = len(np.unique(cluster_labels[picks]))
        return {
            "clusters_hitrate": clusters_hit / K,
            "collisions_rate": (len(picks) - clusters_hit) / K,
        }


    def score_all_humans(human_collections: list[pl.DataFrame], cluster_labels: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame([score_one_human(h, cluster_labels) for h in human_collections])

    return (score_all_humans,)


@app.cell
def _(
    a_labels,
    g_labels,
    human_collections,
    kmeans_labels,
    np,
    pl,
    score_all_humans,
):
    def score_methods(human_collections, methods: dict[str, np.ndarray]) -> pl.DataFrame:
        return pl.concat([
            score_all_humans(human_collections, labels).with_columns(method=pl.lit(name))
            for name, labels in methods.items()
        ])


    clustering_scores = score_methods(
        human_collections,
        {"kmeans": kmeans_labels, "agglomerative": a_labels, "gmm": g_labels },
    )
    clustering_scores.group_by("method").agg(pl.col("clusters_hitrate").mean()) # aggreagate the score from a list of humans
    return clustering_scores, score_methods


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate a random score distribution and average it out
    """)
    return


@app.cell
def _(
    clustering_scores,
    dummy_labels,
    human_collections,
    np,
    pl,
    score_all_humans,
):
    def random_scores(human_collections, label_runs: list[np.ndarray], method_name: str) -> pl.DataFrame:
        return pl.concat([
            score_all_humans(human_collections, labels).with_columns(
                method=pl.lit(method_name)
            )
            for seed, labels in enumerate(label_runs)
        ])

    all_scores = pl.concat([ clustering_scores, random_scores(human_collections, dummy_labels, method_name="shuffled") ])
    return all_scores, random_scores


@app.cell
def _(all_scores, pl):
    all_scores.group_by("method").agg(pl.col("clusters_hitrate").mean(),pl.col("collisions_rate").mean()) # aggreagate the score from a list of humans)
    return


@app.cell
def _(
    a_labels,
    g_labels,
    gmm_pca_labels,
    kmeans_labels,
    kmeans_meta_labels,
    mo,
    time_labels,
):
    method_picker = mo.ui.dropdown(
        options={
            "kmeans": kmeans_labels,
            "agglomerative": a_labels,
            "gmm": g_labels,
            "gmm_pca": gmm_pca_labels,
            "kmeans_meta": kmeans_meta_labels,
            "time_only": time_labels,
        },
        value="kmeans",
        label="method",
    )
    method_picker
    return (method_picker,)


@app.cell
def _():
    return


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

    These results are for Amsterdam, DINOv2 small, k-means with `random_state = 0`.

    The k-means clusters spread the human picks less than random clusters.

    - Clusters 8 and 3 have 29% of the images, but they get 65% of the votes (130 of 200).
    - Clusters 5 and 6 have 18 images. They get no votes.
    - The mean is 2 votes for each image (200 votes, 100 images).

    DINOv2 puts the images that look the same into one cluster.
    The humans pick many images from the clusters with the iconic images, and no images from other clusters.



    Thus, the rule "pick one image from each cluster" does not agree with the humans for this album...................
    .
    """)
    return


@app.cell
def _(method_picker, mo):
    cluster_picker = mo.ui.dropdown(
        options=[str(c) for c in sorted(set(method_picker.value))],
        value="8",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Add photo metadata to the features
    Each photo has a time (`datetaken`) and a GPS position (`latitude`, `longitude`).
    If the humans pick images from all parts of the trip, time clusters can agree better with the humans.

    We add the scaled metadata to the embedding. A weight sets the effect of the metadata.
    """)
    return


@app.cell
def _(np, pl):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    SCALERS = {"standard": StandardScaler}


    def raw_metadata(df: pl.DataFrame) -> np.ndarray:
        return df.select(
            pl.col("datetaken").str.to_datetime().dt.epoch("s").cast(pl.Float64),
            "latitude",
            "longitude",
        ).to_numpy()


    def metadata_features(df: pl.DataFrame, scaler: str, clip: float | None) -> np.ndarray:
        meta = SCALERS[scaler]().fit_transform(raw_metadata(df))
        return meta if clip is None else np.clip(meta, -clip, clip)

    return metadata_features, raw_metadata


@app.cell
def _(np, score_all_humans):
    def corrected_hitrate(humans, labels: np.ndarray, n_runs: int = 200) -> float:
        rng = np.random.default_rng(0)
        real = score_all_humans(humans, labels)["clusters_hitrate"].mean()
        null = np.mean([
            score_all_humans(humans, rng.permutation(labels))["clusters_hitrate"].mean()
            for _ in range(n_runs)
        ])
        return real - null

    return (corrected_hitrate,)


@app.cell
def _(K, KMeans, X, city_emb, metadata_features, np, random_state):
    def meta_kmeans_labels(scaler: str, clip: float | None, weight: float) -> np.ndarray:
        meta = metadata_features(city_emb, scaler, clip)
        return KMeans(n_clusters=K, random_state=random_state).fit_predict(
            np.hstack([X, weight * meta])
        )

    return (meta_kmeans_labels,)


@app.cell
def _(corrected_hitrate, human_collections, meta_kmeans_labels, pl):
    from sklearn.model_selection import ParameterGrid

    _grid = ParameterGrid({
        "scaler": ["standard", "robust", "minmax"],
        "clip": [None, 3.0],
        "weight": [0.0, 0.25, 0.5, 1.0, 2.0],
    })
    _tune, _holdout = human_collections[::2], human_collections[1::2]


    def grid_row(params: dict) -> dict:
        labels = meta_kmeans_labels(**params)
        return params | {
            "tune": corrected_hitrate(_tune, labels),
            "holdout": corrected_hitrate(_holdout, labels),
        }


    meta_grid = pl.DataFrame([grid_row(p) for p in _grid]).sort("tune", descending=True)
    meta_grid
    return (meta_grid,)


@app.cell
def _(
    K,
    KMeans,
    city_emb,
    meta_grid,
    meta_kmeans_labels,
    random_state,
    raw_metadata,
):
    best_meta_params = meta_grid.select("scaler", "clip", "weight").row(0, named=True)
    kmeans_meta_labels = meta_kmeans_labels(**best_meta_params)
    time_labels = KMeans(n_clusters=K, random_state=random_state).fit_predict(
        raw_metadata(city_emb)[:, :1]
    )
    best_meta_params
    return kmeans_meta_labels, time_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fix GMM with PCA
    GMM gives the same clusters as k-means. There are two causes:

    1. The embedding has 384 dimensions, and the album has 100 images.
       A full covariance matrix has about 74,000 parameters for each cluster.
       GMM cannot calculate these parameters from 100 images.
       The regularization (`reg_covar`) controls the result, and GMM operates as spherical k-means.
    2. GMM starts from the k-means result (`init_params="kmeans"`). Then it does not move.

    The fix:

    - We use PCA to decrease the number of dimensions to 10.
      With a full covariance, each cluster then has about 55 parameters.
    - We start each Gaussian at a random image (`init_params="random_from_data"`).
    - We keep the best result of 10 starts (`n_init=10`).

    The adjusted Rand index (ARI) compares two clusterings. An ARI of 1 shows the same clusters.
    Do not compare the label arrays directly. The cluster numbers are arbitrary.
    """)
    return


@app.cell
def _(GaussianMixture, K, PCA, X, g_labels, kmeans_labels, random_state):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.pipeline import make_pipeline


    def pca_gmm(n_dims: int, covariance: str = "full"):
        return make_pipeline(
            PCA(n_components=n_dims, random_state=random_state),
            GaussianMixture(
                n_components=K,
                covariance_type=covariance,
                init_params="random_from_data",
                n_init=10,
                random_state=random_state,
            ),
        )


    gmm_pca_labels = pca_gmm(n_dims=10).fit_predict(X)
    {
        "ARI gmm vs kmeans": adjusted_rand_score(kmeans_labels, g_labels),
        "ARI gmm_pca vs kmeans": adjusted_rand_score(kmeans_labels, gmm_pca_labels),
    }
    return (gmm_pca_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare the new clusterings with the humans
    Each method gets its own shuffled baseline.
    The shuffled baseline keeps the cluster sizes of its method.
    If a method is not better than its own shuffled baseline, its clusters do not agree with the humans.
    """)
    return


@app.cell
def _(
    gmm_pca_labels,
    human_collections,
    kmeans_labels,
    kmeans_meta_labels,
    np,
    pl,
    random_scores,
    score_methods,
    time_labels,
):
    _methods = {
        "kmeans": kmeans_labels,
        "gmm_pca": gmm_pca_labels,
        "kmeans_meta": kmeans_meta_labels,
        "time_only": time_labels,
    }
    _rng = np.random.default_rng(0)
    _shuffled = [
        random_scores(
            human_collections,
            [_rng.permutation(labels) for _ in range(1000)],
            method_name=f"shuffled_{name}",
        )
        for name, labels in _methods.items()
    ]
    (
        pl.concat([score_methods(human_collections, _methods), *_shuffled])
        .group_by("method")
        .agg(
            pl.col("clusters_hitrate").mean().alias("mean"),
            pl.col("clusters_hitrate").std().alias("std"),
        )
        .sort("method")
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
