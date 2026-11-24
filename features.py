from functools import cache
from pathlib import Path

import numpy as np
import polars as pl

MUSIQ_URL = "https://www.kaggle.com/models/google/musiq/tensorFlow2/{variant}/1"


@cache
def _dino_pipe():
    from transformers import pipeline

    return pipeline(task="image-feature-extraction", model="facebook/dinov2-small")


def add_dino_embeddings(
    df: pl.DataFrame, col: str = "path", batch_size: int = 32
) -> pl.DataFrame:
    out = _dino_pipe()(df[col].to_list(), pool=True, batch_size=batch_size)
    emb = np.asarray(out, dtype=np.float32).reshape(len(df), -1)
    return df.with_columns(pl.Series("dino_embeddings", emb))


@cache
def _musiq_fn(variant: str):
    import tensorflow_hub as hub

    return hub.load(MUSIQ_URL.format(variant=variant)).signatures["serving_default"]


def musiq_score(path: str, variant: str) -> float:
    import tensorflow as tf

    img_bytes = tf.constant(Path(path).read_bytes())
    return float(list(_musiq_fn(variant)(img_bytes).values())[0].numpy())


def add_musiq_scores(
    df: pl.DataFrame, variant: str = "koniq-10k", col: str = "path"
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col)
        .map_elements(lambda p: musiq_score(p, variant), return_dtype=pl.Float64)
        .alias("aesthetic")
    )
