import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from dataset_loaders import load_cufed, load_yfcc_city

    return load_cufed, load_yfcc_city, mo, pl


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # YFCC100M city datasets
    Each city is one album of approximately 100 Flickr photos from one owner.
    The cities are Amsterdam, Istanbul, Paris, Tokyo, and Venice.
    The loader downloads missing images to `images/yfcc/<City>/<City>`.
    The loader removes the images that it cannot download.

    **NOTE**: Flickr deleted all the Paris images. Paris shows no images.
    """)
    return


@app.cell
def _(load_cufed, load_yfcc_city, mo):
    from functools import partial

    from dataset_loaders import YFCC_CITIES

    dataset_loaders = {"CUFED": load_cufed} | {
        f"YFCC100M {city}": partial(load_yfcc_city, city) for city in YFCC_CITIES
    }
    dataset_radio = mo.ui.radio(options=dataset_loaders, value="YFCC100M Amsterdam", label="choose a dataset")
    dataset_radio
    return (dataset_radio,)


@app.cell
def _(dataset_radio, mo):
    dataset_full = dataset_radio.value()
    mo.stop(dataset_full.is_empty(), mo.md("**No images found for this dataset.**"))
    return (dataset_full,)


@app.cell
def _(dataset_full):
    dataset_full.describe()
    return


@app.cell
def _(dataset_full):
    dataset_full
    return


@app.cell
def selection(dataset_full, mo, pl):
    selected_album = mo.ui.table(
        dataset_full.group_by("event_id").agg(pl.col("flickr_url").max()),
        selection="single",
        initial_selection=[0],
    )
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
def _(dataset_full, selected_album):
    album = dataset_full.filter(dataset_full["event_id"] == selected_album.value["event_id"])
    return (album,)


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
def _(album):
    album_paths = album.with_columns(
        ("./" + album["path"])
    )
    album_paths["path"]
    return (album_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Embed the dataset
    Embed the images with the selected encoder and store the result in the `embedding` column.
    """)
    return


@app.cell
def _(album_paths, pipe, pl):
    import numpy as np
    from tqdm.auto import tqdm

    paths = album_paths["path"].to_list()

    vecs = []
    for emb in tqdm(pipe(paths, batch_size=10, return_tensors=True), total=len(paths)):
        vecs.append(emb.squeeze(0).detach().cpu().numpy())

    album_emb = album_paths.with_columns(
        pl.Series("embedding", np.stack(vecs))
    )
    return album_emb, np


@app.cell
def _(album_emb):
    album_emb
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
    k = mo.ui.slider(label="k", start=1, stop=20, step=1, value=10)
    k
    return (k,)


@app.cell
def _(album_emb, k):
    from sklearn.cluster import KMeans

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=k.value, random_state=0, n_init="auto").fit(album_emb["embedding"])
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
def _(album_emb, kmeans, np):
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(kmeans.cluster_centers_, album_emb["embedding"])
    print(similarity_matrix.shape)
    iconic_indexes = np.argmax(similarity_matrix, axis=1)
    album_emb[iconic_indexes]
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
def _(album_emb, kmeans, np):
    from sklearn.metrics import silhouette_score

    silhouette_score(np.vstack(album_emb["embedding"].to_numpy()), kmeans.labels_)
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
    album_emb,
    iconic_indexes,
    image_column,
    mo,
):
    _paths = album_emb["path"]
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
def _(album_emb, pl):
    from pathlib import Path

    def path_to_bytes(path: str) -> bytes:
        return Path(path).expanduser().read_bytes()

    img_df = album_emb.with_columns(
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
    album_emb,
    iconic_indexes,
    image_column,
    mo,
    musiq_best,
):
    _paths = album_emb["path"]
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
    album_emb,
    iconic_indexes,
    image_column,
    mo,
    musiq_best,
    simw,
    weighted_best,
):
    _paths = album_emb["path"]
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
def _(img_df_scores_scaled):
    from PIL import Image
    _ex_img1 = Image.open(img_df_scores_scaled[0, "path"])

    for tag_id, value in _ex_img1.getexif():
        tag_name = Image.ExifTags.TAGS.get(tag_id, tag_id) 
        print(tag_name, ":", value)

    print(_ex_img1.format)
    print(_ex_img1.size)
    print(_ex_img1.mode)
    print(_ex_img1.info)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluate the selections against the human summaries
    Only the YFCC100M datasets have human summaries.
    In each city, 20 or more persons picked 10 images from the album.

    We compare three methods. Each method picks one image from each cluster:

    - **centroid**: the image that is nearest to the cluster centroid.
    - **musiq**: the image with the highest MUSIQ score.
    - **weighted**: the image with the highest weighted score. The slider above sets the weight.

    We calculate these metrics for each method:

    - **precision**, **recall**, **F1**: the overlap with each human summary. We give the mean over all the humans.
    - **best F1**: the F1 for the human that agrees most with the method.
    - **vote score**: the sum of the human votes for the picked images, divided by the highest possible sum for the same number of images.

    We also calculate two baselines:

    - **random**: random images. We give the mean of 1000 runs. This is the lower limit.
    - **human**: each human against all the other humans. This is the upper limit.

    Set k to 10 for a fair comparison. Each human summary has 10 images.
    """)
    return


@app.cell
def _(album_emb, mo):
    mo.stop(
        "human_ranks" not in album_emb.columns,
        mo.md("**This dataset has no human summaries. Select a YFCC100M dataset.**"),
    )
    return


@app.cell
def _(np, pl):
    def human_pick_sets(df: pl.DataFrame) -> list[set[int]]:
        ranks = df["human_ranks"].to_list()
        return [
            {row for row, r in enumerate(ranks) if r[h] is not None}
            for h in range(len(ranks[0]))
        ]


    def overlap_scores(picks: set[int], human: set[int]) -> dict:
        hits = len(picks & human)
        precision = hits / len(picks)
        recall = hits / len(human)
        f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}


    def vote_score(votes: np.ndarray, picks: set[int]) -> float:
        best = np.sort(votes)[::-1][: len(picks)].sum()
        return votes[list(picks)].sum() / best


    def evaluate_picks(picks: set[int], humans: list[set[int]], votes: np.ndarray) -> dict:
        per_human = pl.DataFrame([overlap_scores(picks, h) for h in humans])
        return {
            "precision": per_human["precision"].mean(),
            "recall": per_human["recall"].mean(),
            "f1": per_human["f1"].mean(),
            "best_f1": per_human["f1"].max(),
            "vote_score": vote_score(votes, picks),
        }

    return evaluate_picks, human_pick_sets, overlap_scores


@app.cell
def _(album_emb, iconic_indexes, musiq_best, weighted_best):
    _row_of = {p: i for i, p in enumerate(album_emb["path"])}
    method_picks = {
        "centroid": {int(i) for i in iconic_indexes},
        "musiq": {_row_of[p] for p in musiq_best["path"]},
        "weighted": {_row_of[p] for p in weighted_best["path"]},
    }
    return (method_picks,)


@app.cell
def _(evaluate_picks, np, overlap_scores, pl):
    def random_baseline(n_images: int, k: int, humans, votes, n_runs: int = 1000) -> dict:
        rng = np.random.default_rng(0)
        runs = [
            evaluate_picks(set(rng.choice(n_images, size=k, replace=False).tolist()), humans, votes)
            for _ in range(n_runs)
        ]
        return pl.DataFrame(runs).mean().to_dicts()[0]


    def human_baseline(humans: list[set[int]], votes) -> dict:
        f1 = [
            overlap_scores(humans[i], humans[j])["f1"]
            for i in range(len(humans))
            for j in range(len(humans))
            if i != j
        ]
        votes_each = [
            evaluate_picks(h, humans, votes - np.isin(np.arange(len(votes)), list(h)))["vote_score"]
            for h in humans
        ]
        return {"f1": float(np.mean(f1)), "vote_score": float(np.mean(votes_each))}

    return human_baseline, random_baseline


@app.cell
def _(
    album_emb,
    evaluate_picks,
    human_baseline,
    human_pick_sets,
    k,
    method_picks,
    pl,
    random_baseline,
):
    _humans = human_pick_sets(album_emb)
    _votes = album_emb["human_votes"].to_numpy()
    _rows = [
        {"method": name} | evaluate_picks(picks, _humans, _votes)
        for name, picks in method_picks.items()
    ]
    _rows.append({"method": "random"} | random_baseline(len(album_emb), k.value, _humans, _votes))
    _rows.append({"method": "human"} | human_baseline(_humans, _votes))
    evaluation = pl.DataFrame(_rows, strict=False)
    evaluation
    return (evaluation,)


if __name__ == "__main__":
    app.run()
