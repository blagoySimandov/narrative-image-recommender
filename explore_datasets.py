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
    df = pl.read_csv("datasets/yfcmmf00m-cities-amsterdam/98.csv", separator="\t")
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
def _():
    return


if __name__ == "__main__":
    app.run()
