import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from dataset_loaders import YFCC_IMAGES_ROOT
    from pathlib import Path

    return Path, mo, pl


@app.cell
def _(pl):
    df = pl.read_csv("datasets/yfcmmf00m-cities-amsterdam/240.csv", separator="\t")
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.table(df)
    return


@app.cell
def _(Path, pl):
    people_pat = r"(?i)people|person|portrait|\bman\b|\bwoman\b|\bboy\b|\bgirl\b|child|kids|face|couple|friends|tourist|selfie"

    results = []
    for csv_path in sorted(Path("datasets/yfcmmf00m-cities-amsterdam").glob("*.csv")):
        df_inner = pl.read_csv(
            csv_path,
            separator="\t",
            quote_char=None,
            truncate_ragged_lines=True,
            infer_schema_length=0,
        )
        people_df = df_inner.filter(pl.col("tag_ss").str.contains(people_pat))
        results.append({
            "album": csv_path.stem,
            "total": df_inner.height,
            "people": people_df.height,
        })
    return (results,)


@app.cell
def _(mo, results):
    mo.ui.table(results)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The albums with the most people are  98,99,208,96...
    """)
    return


@app.cell
def _(pl):
    df_sis = pl.read_json("datasets/sis/train.story-in-sequence.json")
    df_sis.select(pl.col("annotations").explode()).head(10)
    ann = (
        df_sis.select(pl.col("annotations").explode())
        .select(pl.col("annotations").explode())
        .unnest("annotations")
    )

    print(ann.columns)  # check the field names first
    return ann, df_sis


@app.cell
def _(ann, mo):
    mo.ui.table(ann)
    return


@app.cell
def _(ann, pl):
    stories_sis = (
        ann.sort("story_id", "worker_id", "worker_arranged_photo_order")
        .group_by("story_id", "worker_id", maintain_order=True)
        .agg(
            pl.col("album_id").first(),
            pl.col("text").alias("sentences"),
            pl.col("photo_flickr_id").alias("photos"),
        )
    )
    return (stories_sis,)


@app.cell
def _(mo, stories_sis):
    table = mo.ui.table(stories_sis.head(200), selection="single", page_size=10)
    table
    return (table,)


@app.cell
def _(df_sis, pl):
    imgs = df_sis.select(pl.col("images").explode()).unnest("images")
    print(imgs.columns)  # check which url fields exist
    url_by_id = dict(zip(imgs["id"], imgs["url_o"]))
    return (url_by_id,)


@app.cell
def _(mo, pl, stories_sis, table, url_by_id):
    mo.stop(len(table.value) == 0, mo.md("_Select a row above_"))

    row = table.value.row(0, named=True)

    def story_block(r, n):
        photos = [
            mo.image(src=url_by_id[p], width=180, height=140, rounded=True)
            if p in url_by_id else mo.md("_no image_")
            for p in r["photos"]
        ]
        captions = [mo.md(t) for t in r["sentences"]]
        return mo.vstack([
            mo.md("---"),
            mo.md(f"#### Story {n} &nbsp; <small>story_id {r['story_id']}</small>"),
            mo.hstack(photos, justify="start", gap=1),
            mo.hstack(captions, justify="start", gap=1, align="start"),
        ])

    album_stories = stories_sis.filter(pl.col("album_id") == row["album_id"])

    mo.vstack([
        mo.md(f"### Album {row['album_id']}: {album_stories.height} stories"),
        *[story_block(r, n) for n, r in enumerate(album_stories.iter_rows(named=True), 1)],
    ], gap=1)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
