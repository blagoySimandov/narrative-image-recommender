---
title: Compare
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import polars as pl

from dataset_loaders import load_yfcc_city
from evaluation import comparison_chart, human_summaries, normalise, score_systems, summarise
from features import add_dino_embeddings, add_musiq_scores
from systems import select_full_system, select_graph_path
```

# Compare the systems

Every system gets the same photos and the same k.
A system is a function that returns an ordered list of `image_id`.
To add a system, add one entry to the `systems` dict below.

```python {.marimo}
city_name = "amsterdam"
k = 10
city = load_yfcc_city(city_name)
```

The full system uses the `koniq-10k` MUSIQ model. The graph uses the `ava` model.

```python {.marimo}
with mo.persistent_cache(name=f"features-{city_name}"):
    embedded = add_dino_embeddings(city)
    df_koniq = add_musiq_scores(embedded, variant="koniq-10k")
    df_ava = add_musiq_scores(embedded, variant="ava")
```

```python {.marimo}
def ids(df: pl.DataFrame) -> list[str]:
    return df["image_id"].to_list()


systems = {
    "Top-k aesthetic": ids(df_koniq.sort("aesthetic", descending=True).head(k)),
    "Clustering and Aesthetic Selection": ids(select_full_system(df_koniq, k)),
    **{f"Graph time b={b}": ids(select_graph_path(df_ava, k, budget=b, by_time=True)) for b in (3, 5, 7)},
}
systems
```

```python {.marimo}
scores = score_systems(systems, city_name, all_ids=city["image_id"].to_numpy())
summary = summarise(scores)
summary
```

`normalised` = (system − random) / (human − random).
0 means no better than random. 1 means as good as humans agree with each other.

```python {.marimo}
normalise(summary)
```

# Scoring

```python {.marimo}
mo.ui.altair_chart(comparison_chart(scores))
```

## Selected paths

Each row is one ordered selection.
The human row is the first human summary.

```python {.marimo}
paths = city.select("image_id", "path").to_dicts()
path_of = {r["image_id"]: r["path"] for r in paths}


def strip(label: str, image_ids) -> mo.Html:
    thumbs = [mo.image(src=path_of[i], width=140, height=140) for i in image_ids]
    return mo.vstack([mo.md(f"**{label}**"), mo.hstack(thumbs, justify="start")])


mo.vstack(
    [
        strip("Clustering and Aesthetic Selection", systems["Clustering and Aesthetic Selection"]),
        strip("Graph time b=3", systems["Graph time b=3"]),

        strip("Human 1", human_summaries(city_name, k)[0]),
    ]
)
```

```python {.marimo}

```