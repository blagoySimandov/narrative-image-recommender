import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path

    return Path, mo


@app.cell
def _(Path):
    BASE = "./datasets/"
    parquet_path = Path(BASE,'instagram.parquet')
    return (parquet_path,)


@app.cell
def _(mo, parquet_path):
    _df = mo.sql(
        f"""
        SELECT
          query,
          username,
          followers,
          p.code,
          p.url AS post_url,
          p.likes,
          p.comments,
          img_idx,
          img,
        img.url as url 
        FROM read_parquet('{parquet_path}') AS prof,
          unnest(prof.posts) AS t(p),
          unnest(p.images) WITH ORDINALITY AS i(img, img_idx),
        """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
