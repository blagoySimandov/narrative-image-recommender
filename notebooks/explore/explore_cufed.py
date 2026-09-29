import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CUFED Test Split

    This notebook examines the CUFED test split. The split has 133 albums and
    5,665 images. Each album has one or two event labels. Each image has one
    human importance score.

    The importance score gives the value of an image in its album. The score
    is not a measure of technical quality. A blurred image of an important
    moment can have a high score.

    Source: https://huggingface.co/datasets/Shawn-Huang/CUFED-AlbumBench.
    The license is CC BY-NC 2.0. Use the data only for non-commercial research.
    """)
    return


@app.cell
def _():
    import altair as alt
    import polars as pl

    from dataset_loaders import load_cufed
    from image_pipeline import encode_images, load_clip

    return alt, encode_images, load_clip, load_cufed, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the Data

    The loader joins the three source files into one table. Each row is one
    image. The `event_types` column is a list, because 34 albums have two
    event labels.
    """)
    return


@app.cell
def _(load_cufed):
    cufed = load_cufed()
    cufed
    return (cufed,)


@app.cell
def _(cufed, pl):
    albums = cufed.group_by("event_id").agg(
        pl.len().alias("images"),
        pl.col("event_types").first(),
        pl.col("context_importance").mean().alias("mean_importance"),
        pl.col("context_importance").std().alias("std_importance"),
    )
    albums.select("images", "mean_importance", "std_importance").describe()
    return (albums,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Event Classes

    This chart shows the number of albums for each event class. An album with
    two labels counts one time in each class.
    """)
    return


@app.cell
def _(albums, alt, mo, pl):
    event_counts = (
        albums.explode("event_types", empty_as_null=True)
        .group_by("event_types")
        .agg(pl.len().alias("albums"))
        .rename({"event_types": "event_type"})
    )
    mo.ui.altair_chart(
        alt.Chart(event_counts)
        .mark_bar()
        .encode(
            x=alt.X("albums:Q", title="Albums"),
            y=alt.Y("event_type:N", sort="-x", title=None),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importance Scores

    The scores are in the range -2 to 2. The scores are averages of a small
    number of ratings, so the values are discrete. Other studies report an
    agreement of about 0.4 (Spearman) between two groups of annotators. This
    value sets a limit on the agreement that a model can get.
    """)
    return


@app.cell
def _(alt, cufed, mo):
    mo.ui.altair_chart(
        alt.Chart(cufed)
        .mark_bar()
        .encode(
            x=alt.X("context_importance:Q", bin=alt.Bin(step=0.25), title="Importance"),
            y=alt.Y("count():Q", title="Images"),
        )
    )
    return


@app.cell
def _(cufed, pl):
    (
        cufed.explode("event_types", empty_as_null=True)
        .group_by("event_types")
        .agg(
            pl.len().alias("images"),
            pl.col("context_importance").mean().alias("mean"),
            (pl.col("context_importance") > 0).mean().alias("share_positive"),
        )
        .sort("mean", descending=True)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Album Browser

    Select an album. The grid shows the images from the highest score to the
    lowest score.
    """)
    return


@app.cell
def _(albums, mo):
    album_options = {
        f"{r['event_id']} ({', '.join(r['event_types'])}, {r['images']})": r["event_id"]
        for r in albums.sort("event_id").iter_rows(named=True)
    }
    album_picker = mo.ui.dropdown(
        options=album_options, value=next(iter(album_options)), label="Album"
    )
    album_picker
    return (album_picker,)


@app.cell
def _(album_picker, cufed, pl):
    album = cufed.filter(pl.col("event_id") == album_picker.value).sort(
        "context_importance", descending=True
    )
    return (album,)


@app.cell
def _(mo):
    def image_card(row: dict, label: str):
        return mo.vstack(
            [mo.image(src=row["path"], width=160), mo.md(f"<small>{label}</small>")],
            align="center",
        )

    def image_grid(rows: list[dict], label_of):
        cards = [image_card(r, label_of(r)) for r in rows]
        return mo.hstack(cards, wrap=True, justify="start", gap=0.5)

    return (image_grid,)


@app.cell
def _(album, image_grid):
    image_grid(
        album.to_dicts(), lambda r: f"{r['image_id']} · {r['context_importance']:+.2f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weak Shot Groups

    CUFED has no burst labels. The images have no EXIF data, so the dataset
    gives no capture times. This section makes weak shot groups from CLIP
    image embeddings only.

    Two images go into the same group if their cosine similarity is equal to
    or more than the threshold. The groups are the connected components of
    this graph.

    Use the groups for exploration only. They are not ground truth for the
    best frame in a burst.
    """)
    return


@app.cell
def _(load_clip):
    processor, model, device = load_clip()
    return device, model, processor


@app.cell
def _(album, device, encode_images, model, processor):
    album_embeddings = encode_images(album["path"].to_list(), processor, model, device)
    return (album_embeddings,)


@app.cell
def _(mo):
    threshold = mo.ui.slider(0.80, 0.99, step=0.01, value=0.92, label="Threshold")
    threshold
    return (threshold,)


@app.function
def connected_groups(similarity, threshold: float) -> list[int]:
    n = similarity.shape[0]
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                parent[root(i)] = root(j)
    return [root(i) for i in range(n)]


@app.cell
def _(album, album_embeddings, pl, threshold):
    similarity = (album_embeddings @ album_embeddings.T).cpu().numpy()
    grouped = album.with_columns(
        pl.Series("shot_group_id", connected_groups(similarity, threshold.value))
    )
    groups = (
        grouped.group_by("shot_group_id")
        .agg(
            pl.len().alias("size"),
            pl.col("context_importance").max().alias("max_importance"),
            (
                pl.col("context_importance").max() - pl.col("context_importance").min()
            ).alias("importance_range"),
        )
        .filter(pl.col("size") > 1)
        .sort("size", descending=True)
    )
    groups
    return grouped, groups


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The table above shows each group with more than one image. A large
    `importance_range` shows that the annotators gave different scores to
    near-duplicate images. The grids below show each group.
    """)
    return


@app.cell
def _(grouped, groups, image_grid, mo, pl):
    mo.stop(groups.height == 0, mo.md("_No group has more than one image._"))
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(f"**Group {g['shot_group_id']}**: {g['size']} images"),
                    image_grid(
                        grouped.filter(pl.col("shot_group_id") == g["shot_group_id"])
                        .sort("context_importance", descending=True)
                        .to_dicts(),
                        lambda r: f"{r['context_importance']:+.2f}",
                    ),
                ]
            )
            for g in groups.iter_rows(named=True)
        ],
        gap=1,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
