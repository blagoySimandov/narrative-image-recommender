import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from dataset_loaders import load_cufed
    import torch

    return load_cufed, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cufed dataset
    Created by mechanical turk people. 3 people labeled the type of location of the album - we don't use this

    Has importance score + label of location.

    Importance score tries to measure how important an image is to identify the event.

    5 workers labeled the importance score. It was then averaged.
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
def _():
    return


@app.cell
def _(cufed_dataset_full, selected_album):
    filtered_cufed = cufed_dataset_full.filter(cufed_dataset_full["event_id"] ==selected_album.value["event_id"])
    return (filtered_cufed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Select encodeer
    We are using huggingface to embedd the images.
    We are oging to use a marimo radio button to select between and test out differetn encoders


    **NOTE**: To use DiNOv3 you need to request access from META. They usually approve fast.
    """)
    return


@app.cell
def _(mo):
    radiogroup = mo.ui.radio(
        options=["facebook/dinov2-small-imagenet1k-1-layer","facebook/dinov3-vith16plus-pretrain-lvd1689m"], value="facebook/dinov2-small-imagenet1k-1-layer", label="choose one"
    )

    radiogroup
    return (radiogroup,)


@app.cell
def _(radiogroup):
    from transformers import pipeline


    pipe = pipeline(
        task="image-feature-extraction",
       model=radiogroup.value,
        pool=True,
        device=0
    )

    #pipe(url)
    return (pipe,)


@app.cell
def _(filtered_cufed):
    cd_with_paths = filtered_cufed.with_columns(
        ("./" + filtered_cufed["path"])
    )
    cd_with_paths["path"]
    return (cd_with_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Embed the dataset using dingov2
    Embed + store the embeddings in the embedding column
    """)
    return


@app.cell
def _(cd_with_paths, pipe, pl):
    import numpy as np
    from tqdm.auto import tqdm

    paths = cd_with_paths["path"].to_list()

    vecs = []
    for emb in tqdm(pipe(paths, batch_size=10, return_tensors=True), total=len(paths)):
        vecs.append(emb.squeeze(0).detach().cpu().numpy())

    cde = cd_with_paths.with_columns(
        pl.Series("embedding", np.stack(vecs))
    )
    return cde, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell
def _(cde):
    cde
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Create the algorithm for Max Coverage
    Run K means to cluster the photos' embeddings and find the "inconic images".
    First select k
    """)
    return


@app.cell
def _(mo):
    k = mo.ui.slider(label="k",start=1, stop=10, step=1, value=5) # default to 5k
    k
    return (k,)


@app.cell
def _(cde, k):
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#:~:text=the%20training%20set.-,Examples,-%3E%3E%3E%20from%20sklearn

    kmeans = KMeans(n_clusters=k.value, random_state=0, n_init="auto").fit(cde["embedding"])
    return cosine_similarity, kmeans


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pick the iconic imags
    Currently we pick the iconic images by selecting the ones wich best cover the dataset.
    In our case these would be the images that are closest to each centroid.



    Go through the K cluster centers and similarity match them to the dataset
    """)
    return


@app.cell
def _(cde, cosine_similarity, kmeans, np):
    similarity_matrix = cosine_similarity(kmeans.cluster_centers_, cde["embedding"])
    print(similarity_matrix.shape)
    best_indexes = np.argmax(similarity_matrix,axis=1)
    cde[best_indexes]
    return (best_indexes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # See the silhoute score

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
def _(cde, kmeans, np):
    from sklearn.metrics import silhouette_score
    silhouette_score(np.vstack(cde["embedding"].to_numpy()), kmeans.labels_)
    return


@app.cell
def _(best_indexes, kmeans):
    from collections import defaultdict

    grouped_images = defaultdict(list)

    for l in range(len(kmeans.labels_)):
        cluster_id = kmeans.labels_[l] 
        iconic_image = best_indexes[cluster_id]
        grouped_images[iconic_image].append(l)
    return (grouped_images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Showcase the groupings
    Code in the cell below was fully generated by AI
    """)
    return


@app.cell
def _(cde, grouped_images, kmeans, mo):
    marimo_output = []

    for gi in sorted(grouped_images.keys()):
        iconic_row = cde.row(gi, named=True)
        cluster_id_new = kmeans.labels_[gi]
        header = mo.md(f"### 🌟 Cluster {cluster_id_new} (Iconic Image Index: {gi})")
        iconic_img_ui = mo.vstack([
            mo.md("**Iconic Match:**"),
            mo.image(src=iconic_row["path"], width=180, height=180)
        ], align="center")
        member_images = []
        for member_idx in grouped_images[gi]:
            member_row = cde.row(member_idx, named=True)
            img = mo.image(src=member_row["path"], width=100, height=100)
            member_images.append(img)
        members_gallery = mo.vstack([
            mo.md("**Cluster Members:**"),
            mo.hstack(member_images, wrap=True, gap=1)
        ])
        cluster_block = mo.vstack([
            header,
            mo.hstack([iconic_img_ui, members_gallery], gap=3, align="start"),
            mo.md("---") # Visual separator
        ])
        marimo_output.append(cluster_block)
    mo.vstack(marimo_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Eval
    Let's now match the iconic images to their importance as labeled by the cufed dataset
    """)
    return


@app.cell
def _(cde, k, pl):
    selected = cde.select(
        pl.all().top_k_by("context_importance", k.value)
    )
    selected
    return (selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now take their sum and compare to the sum of our selection
    """)
    return


@app.cell
def _(selected):
    importance_score_label = selected.sum()["context_importance"]
    return (importance_score_label,)


@app.cell
def _(best_indexes, cde):
    importance_score_kmeans = cde[best_indexes].sum()['context_importance']
    return (importance_score_kmeans,)


@app.cell
def _(importance_score_kmeans, importance_score_label):
    score =   importance_score_kmeans/ importance_score_label
    score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Picking Iconic Images via Aesthetic scoring each cluster
    After we generate the clusters using KMeans a better way to pick each iconic image would be via an aesthetic scorer
    We can run it on the clustered images.
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
    predict_fn = model.signatures['serving_default']
    return predict_fn, tf


@app.cell
def _(cde, pl):
    from pathlib import Path
    def path_to_bytes(path: str) -> bytes:
        return Path(path).expanduser().read_bytes()
    img_df = cde.with_columns(
        (pl.col("path").map_elements(path_to_bytes,return_dtype=pl.Binary).alias("img_bytes"))
    )

    return (img_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Find the most aesthetic photo in each cluster
    Use MUSIQ from Google to find the most aesthetic photo from each cluster
    """)
    return


@app.cell
def _(img_df, kmeans, pl, predict_fn, tf):
    def score_bytes(b: bytes) -> float:
        return float(list(predict_fn(tf.constant(b)).values())[0].numpy())
    
    img_df_scores = img_df.with_columns(
        pl.col("img_bytes").map_elements(score_bytes, return_dtype=pl.Float64).alias("musiq_score"),
        pl.Series("cluster", kmeans.labels_).cast(pl.Int64)
    )
    

    return (img_df_scores,)


@app.cell
def _(img_df_scores):
    best = (
        img_df_scores
        .sort("musiq_score", descending=True)
        .group_by("cluster", maintain_order=True)
        .first()
        .sort("cluster")
    )
    best
    return (best,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Code below is AI generated<<<
    """)
    return


@app.cell
def _(best, cde, grouped_images, kmeans, mo):
    # cluster id -> iconic index (key of grouped_images)
    _gi_by_cluster = {int(kmeans.labels_[_gi]): _gi for _gi in grouped_images}

    _blocks = []

    for _r in best.sort("musiq_score", descending=True).iter_rows(named=True):
        _cluster_id = int(_r["cluster"])
        _gi = _gi_by_cluster.get(_cluster_id)
        if _gi is None:
            continue

        _header = mo.md(
            f"### ⭐ Cluster {_cluster_id}: MUSIQ {_r['musiq_score']:.1f}"
        )

        _best_ui = mo.vstack([
            mo.md("**MUSIQ Match:**"),
            mo.image(src=_r["path"], width=180, height=180),
        ], align="center")

        _iconic_row = cde.row(_gi, named=True)
        _iconic_ui = mo.vstack([
            mo.md(f"*Centroid-closest (idx {_gi})*"),
            mo.image(src=_iconic_row["path"], width=100, height=100),
        ], align="center")

        _member_images = [
            mo.image(src=cde.row(_i, named=True)["path"], width=100, height=100)
            for _i in grouped_images[_gi]
        ]

        _members_gallery = mo.vstack([
            mo.md(f"**Cluster Members:** ({len(_member_images)})"),
            mo.hstack(_member_images, wrap=True, gap=1),
        ])

        _blocks.append(mo.vstack([
            _header,
            mo.hstack([_best_ui, _iconic_ui, _members_gallery], gap=3, align="start"),
            mo.md("---"),
        ]))

    mo.vstack(_blocks)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
