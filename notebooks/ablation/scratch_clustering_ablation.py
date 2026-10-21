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
    return X2, plt


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
    return (clustering_scores,)


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
    return (all_scores,)


@app.cell
def _(all_scores, pl):
    all_scores.group_by("method").agg(pl.col("clusters_hitrate").mean(),pl.col("collisions_rate").mean()) # aggreagate the score from a list of humans)
    return


@app.cell
def _(a_labels, g_labels, kmeans_labels, mo):
    method_picker = mo.ui.dropdown(
        options={"kmeans": kmeans_labels, "agglomerative": a_labels, "gmm": g_labels},
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
        tiles = [
            mo.vstack([mo.image(src=r["path"], width=160, height=160), mo.md(f"votes: **{r['human_votes']}**")], align="center")
            for r in members.iter_rows(named=True)
        ]
        return mo.vstack([
            mo.md(f"### Cluster {cluster}: {len(members)} images, {members['human_votes'].sum()} votes"),
            mo.hstack(tiles, wrap=True, gap=1),
        ])


    return (cluster_gallery,)


@app.cell
def _(city_emb, cluster_gallery, cluster_picker, method_picker):
    cluster_gallery(city_emb, method_picker.value, int(cluster_picker.value))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
