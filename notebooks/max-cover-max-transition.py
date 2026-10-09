import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from dataset_loaders import load_cufed

    return load_cufed, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CUFED dataset
    Mechanical Turk workers created the labels. 3 workers labeled the location type of each album. We do not use this label.

    Each image has an importance score and a location label.

    The importance score shows how important an image is to identify the event.

    5 workers labeled the importance score. The dataset gives the average.
    """)
    return


@app.cell
def _(load_cufed):
    cufed_dataset_full = load_cufed()
    cufed_dataset_full["context_importance"].describe()
    return (cufed_dataset_full,)


@app.cell
def _(cufed_dataset_full):
    cufed_dataset_full
    return


@app.cell
def selection(cufed_dataset_full, mo, pl):
    selected_album = mo.ui.table(cufed_dataset_full.group_by("event_id").agg(pl.col("flickr_url").max()), selection="single")
    selected_album
    return (selected_album,)


@app.cell
def _(mo, selected_album):
    mo.stop(len(selected_album.value) is 0)
    return


@app.cell
def _():
    return


@app.cell
def _(cufed_dataset_full, selected_album):
    album_cufed = cufed_dataset_full.filter(cufed_dataset_full["event_id"] == selected_album.value["event_id"])
    return (album_cufed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Select encoder
    We use Hugging Face to embed the images.
    Use the marimo radio button to select an encoder and compare different encoders.


    **NOTE**: To use DINOv3, request access from Meta. Meta usually approves fast.
    """)
    return


@app.cell
def _(mo):
    encoders = [
        "facebook/dinov2-small-imagenet1k-1-layer",
        "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    ]
    encoder_radio = mo.ui.radio(options=encoders, value=encoders[0], label="choose one")
    encoder_radio
    return (encoder_radio,)


@app.cell
def _(encoder_radio):
    from transformers import pipeline

    pipe = pipeline(
        task="image-feature-extraction",
        model=encoder_radio.value,
        pool=True,
        device=0,
    )
    return (pipe,)


@app.cell
def _(album_cufed):
    cufed_paths = album_cufed.with_columns(
        ("./" + album_cufed["path"])
    )
    cufed_paths["path"]
    return (cufed_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Embed the dataset
    Embed the images with the selected encoder and store the result in the `embedding` column.
    """)
    return


@app.cell
def _(cufed_paths, pipe, pl):
    import numpy as np
    from tqdm.auto import tqdm

    paths = cufed_paths["path"].to_list()

    vecs = []
    for emb in tqdm(pipe(paths, batch_size=10, return_tensors=True), total=len(paths)):
        vecs.append(emb.squeeze(0).detach().cpu().numpy())

    cufed_emb = cufed_paths.with_columns(
        pl.Series("embedding", np.stack(vecs))
    )
    return cufed_emb, np


@app.cell
def _(cufed_emb):
    cufed_emb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Max coverage algorithm
    Run k-means on the image embeddings to find the "iconic images".
    First, select k.
    """)
    return


@app.cell
def _(mo):
    k = mo.ui.slider(label="k", start=1, stop=10, step=1, value=5)
    k
    return (k,)


@app.cell
def _(cufed_emb, k):
    from sklearn.cluster import KMeans

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=k.value, random_state=0, n_init="auto").fit(cufed_emb["embedding"])
    return (kmeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pick the iconic images
    We pick the iconic images that cover the dataset best.
    These are the images closest to each centroid.

    For each of the k cluster centers, find the most similar image in the dataset.
    """)
    return


@app.cell
def _(cufed_emb, kmeans, np):
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(kmeans.cluster_centers_, cufed_emb["embedding"])
    print(similarity_matrix.shape)
    iconic_indexes = np.argmax(similarity_matrix, axis=1)
    cufed_emb[iconic_indexes]
    return iconic_indexes, similarity_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Silhouette score

    **From the docs**


    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (a) and the mean nearest-cluster distance (b) for each
    sample.  The Silhouette Coefficient for a sample is (b - a) / max(a,
    b).  To clarify, b is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 <= n_labels <= n_samples - 1.

    This function returns the mean Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`silhouette_samples`.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.
    """)
    return


@app.cell
def _(cufed_emb, kmeans, np):
    from sklearn.metrics import silhouette_score

    silhouette_score(np.vstack(cufed_emb["embedding"].to_numpy()), kmeans.labels_)
    return


@app.cell
def _(kmeans):
    from collections import defaultdict

    cluster_members = defaultdict(list)

    for image_idx, cluster_id in enumerate(kmeans.labels_):
        cluster_members[int(cluster_id)].append(image_idx)
    return (cluster_members,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Show the groups
    The helper functions in the next cell make the galleries. AI generated the first version of this code.
    """)
    return


@app.cell
def _(mo):
    def image_column(title, path, size):
        return mo.vstack([mo.md(title), mo.image(src=path, width=size, height=size)], align="center")

    def cluster_block(header, featured, member_paths):
        member_images = [mo.image(src=path, width=100, height=100) for path in member_paths]
        members_gallery = mo.vstack([
            mo.md(f"**Cluster Members:** ({len(member_images)})"),
            mo.hstack(member_images, wrap=True, gap=1),
        ])
        return mo.vstack([
            mo.md(header),
            mo.hstack([*featured, members_gallery], gap=3, align="start"),
            mo.md("---"),
        ])

    return cluster_block, image_column


@app.cell
def _(
    cluster_block,
    cluster_members,
    cufed_emb,
    iconic_indexes,
    image_column,
    mo,
):
    _paths = cufed_emb["path"]
    _blocks = []

    for _cluster_id, _members in sorted(cluster_members.items()):
        _iconic_idx = int(iconic_indexes[_cluster_id])
        _blocks.append(cluster_block(
            f"### Cluster {_cluster_id} (Iconic Image Index: {_iconic_idx})",
            [image_column("**Iconic Match:**", _paths[_iconic_idx], 180)],
            _paths.gather(_members).to_list(),
        ))

    mo.vstack(_blocks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pick iconic images with aesthetic scores
    After k-means makes the clusters, an aesthetic scorer can be a better way to pick each iconic image.
    We run the scorer on the images in each cluster.
    """)
    return


@app.cell
def _():
    # code was "inspired" from
    # https://www.kaggle.com/code/neocosmliang/image-aesthetic-scoring-with-musiq-models
    import tensorflow as tf
    import tensorflow_hub as hub

    model_handle = "https://www.kaggle.com/models/google/musiq/tensorFlow2/koniq-10k/1"
    model = hub.load(model_handle)
    predict_fn = model.signatures["serving_default"]
    return predict_fn, tf


@app.cell
def _(cufed_emb, pl):
    from pathlib import Path

    def path_to_bytes(path: str) -> bytes:
        return Path(path).expanduser().read_bytes()

    img_df = cufed_emb.with_columns(
        pl.col("path").map_elements(path_to_bytes, return_dtype=pl.Binary).alias("img_bytes")
    )
    return (img_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Find the most aesthetic photo in each cluster
    Use MUSIQ from Google to find the most aesthetic photo in each cluster.
    """)
    return


@app.cell
def _(img_df, kmeans, np, pl, predict_fn, similarity_matrix, tf):
    def score_bytes(b: bytes) -> float:
        return float(list(predict_fn(tf.constant(b)).values())[0].numpy())

    img_df_scores = img_df.with_columns(
        pl.col("img_bytes").map_elements(score_bytes, return_dtype=pl.Float64).alias("musiq_score"),
        pl.Series("cluster", kmeans.labels_).cast(pl.Int64),
        pl.Series("centroid_similarity", similarity_matrix[kmeans.labels_, np.arange(len(kmeans.labels_))]),
    )
    return (img_df_scores,)


@app.cell
def _(img_df_scores):
    musiq_best = (
        img_df_scores
        .sort("musiq_score", descending=True)
        .group_by("cluster", maintain_order=True)
        .first()
        .sort("cluster")
    )
    musiq_best
    return (musiq_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Show each cluster with its MUSIQ match and its centroid-closest image.
    """)
    return


@app.cell
def _(
    cluster_block,
    cluster_members,
    cufed_emb,
    iconic_indexes,
    image_column,
    mo,
    musiq_best,
):
    _paths = cufed_emb["path"]
    _blocks = []

    for _row in musiq_best.sort("musiq_score", descending=True).iter_rows(named=True):
        _cluster_id = _row["cluster"]
        _iconic_idx = int(iconic_indexes[_cluster_id])
        _blocks.append(cluster_block(
            f"### Cluster {_cluster_id}: MUSIQ {_row['musiq_score']:.1f}",
            [
                image_column("**MUSIQ Match:**", _row["path"], 180),
                image_column(f"*Centroid-closest (idx {_iconic_idx})*", _paths[_iconic_idx], 100),
            ],
            _paths.gather(cluster_members[_cluster_id]).to_list(),
        ))

    mo.vstack(_blocks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Weigh the aesthetic score against the similarity to the centroid

    $$
    \text{score} = w_1 \cdot \text{sim} + w_2 \cdot \text{aesthetic}
    $$

    We pick the image with the highest score in each cluster.
    """)
    return


@app.cell
def _(mo):
    simw = mo.ui.slider(label="similarity weight", start=0.0, stop=1.0, step=0.01, value=0.5)
    return (simw,)


@app.cell
def _(img_df_scores, pl):
    from sklearn.preprocessing import MinMaxScaler
    cols_to_normalize = ["centroid_similarity", "musiq_score"]
    scaled = MinMaxScaler().fit_transform(img_df_scores.select(cols_to_normalize).to_numpy())

    img_df_scores_scaled = img_df_scores.with_columns(
        pl.Series("centroid_similarity_scaled", scaled[:, 0]),
        pl.Series("musiq_score_scaled", scaled[:, 1])
    )
    img_df_scores_scaled.select("centroid_similarity_scaled","musiq_score_scaled")

    return (img_df_scores_scaled,)


@app.cell
def _(img_df_scores_scaled, pl, simw):
    def weighted_score(similarity_weight: float) -> pl.Expr:
        return (
            similarity_weight * pl.col("centroid_similarity_scaled")
            + (1-similarity_weight) * pl.col("musiq_score_scaled")
        ).alias("weighted_score")


    img_df_weighted = img_df_scores_scaled.with_columns(
        weighted_score(simw.value)
    )
    return (img_df_weighted,)


@app.cell
def _(img_df_weighted):
    cleaned = img_df_weighted.select("flickr_url", "weighted_score","cluster","path") # makes it faster
    weighted_best = (
            cleaned
            .sort("weighted_score", descending=True)
            .group_by("cluster", maintain_order=True)
            .first()
            .sort("cluster")
        )
    weighted_best
    return (weighted_best,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Code below is ai generated
    """)
    return


@app.cell
def _(
    cluster_block,
    cluster_members,
    cufed_emb,
    iconic_indexes,
    image_column,
    mo,
    musiq_best,
    simw,
    weighted_best,
):
    _paths = cufed_emb["path"]
    _musiq_paths = dict(zip(musiq_best["cluster"], musiq_best["path"]))
    _blocks = []

    for _row in weighted_best.sort("weighted_score", descending=True).iter_rows(named=True):
        _cluster_id = _row["cluster"]
        _iconic_idx = int(iconic_indexes[_cluster_id])
        _blocks.append(cluster_block(
            f"### Cluster {_cluster_id}: weighted {_row['weighted_score']:.2f}",
            [
                image_column("**Weighted Match:**", _row["path"], 180),
                image_column("*MUSIQ match*", _musiq_paths[_cluster_id], 100),
                image_column(f"*Centroid-closest (idx {_iconic_idx})*", _paths[_iconic_idx], 100),
            ],
            _paths.gather(cluster_members[_cluster_id]).to_list(),
        ))

    mo.vstack([simw, *_blocks])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
