---
title: Similarity Checking
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import polars as pl
import dataset_loaders
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import zlib
```

```python {.marimo}
amsterdam= dataset_loaders.load_yfcc_city("amsterdam")
table = mo.ui.table(amsterdam, selection="multi")
table
```

```python {.marimo}
selected = table.value
```

```python {.marimo}
from transformers import pipeline

dino_pipe = pipeline(task="image-feature-extraction", model="facebook/dinov2-small")

def add_dino_embeddings(
    df: pl.DataFrame, col: str, batch_size: int = 32
) -> pl.DataFrame:
    out = dino_pipe(df[col].to_list(), pool=True, batch_size=batch_size)
    emb = np.asarray(out, dtype=np.float32).reshape(len(df), -1)
    return df.with_columns(pl.Series("dino_embeddings", emb))
```

```python {.marimo}
embed = add_dino_embeddings(amsterdam,"path")
```

# Compare _cosine_ similarity with compression similarity and PCA sim

```python {.marimo}
from sklearn.metrics.pairwise import cosine_similarity

X = np.stack(embed["dino_embeddings"].to_numpy())

def cosine_sim(X):
    return cosine_similarity(X)

def pca_sim(X, k=32):
    k = min(k, X.shape[0], X.shape[1])
    return cosine_similarity(PCA(n_components=k).fit_transform(X))

def compression_sim(paths, size=(128, 128)):
    imgs = [np.asarray(Image.open(p).convert("L").resize(size)).tobytes() for p in paths]
    c = [len(zlib.compress(b)) for b in imgs]
    n = len(imgs)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cij = len(zlib.compress(imgs[i] + imgs[j]))
            S[i, j] = 1 - (cij - min(c[i], c[j])) / max(c[i], c[j])
    return S
```

```python {.marimo}
pcasim = pca_sim(X)
comp_sim = compression_sim(embed["path"].to_list())
cos_sim = cosine_sim(X)
```

```python {.marimo}
pcasim
```

```python {.marimo}
query = mo.ui.slider(0, len(embed) - 1, label="query photo")
query
```

```python {.marimo}
paths = embed["path"].to_list()
i = query.value

def top_k(S, i, k=5):
    return [j for j in np.argsort(-S[i]) if j != i][:k]

def row(name, S):
    imgs = [mo.vstack([mo.image(paths[j], width=150), mo.md(f"{S[i, j]:.2f}")])
            for j in top_k(S, i)]
    return mo.vstack([mo.md(f"**{name}**"), mo.hstack(imgs, justify="start")])

mo.vstack([
    mo.image(paths[i], width=250),
    row("cosine", cos_sim),
    row("PCA", pcasim),
    row("compression", comp_sim),
])
```

# Places where PCA and cosine disagree the most

```python {.marimo}
def topk_set(S, i, k=5):
    return set(top_k(S, i, k))

overlap = np.array([len(topk_set(cos_sim, i) & topk_set(pcasim, i)) / 5
                    for i in range(len(paths))])
print("average agreement:", overlap.mean())
print("most different queries:", np.argsort(overlap)[:10])
```

```python {.marimo}
a = np.asarray(Image.open(paths[0]).convert("L").resize((128, 128)))

def sim(x, y):
    x, y = x.tobytes(), y.tobytes()
    cx, cy, cxy = len(zlib.compress(x)), len(zlib.compress(y)), len(zlib.compress(x + y))
    return 1 - (cxy - min(cx, cy)) / max(cx, cy)

print("same image:     ", sim(a, a))
print("shifted 1 pixel:", sim(a, np.roll(a, 1, axis=1)))
print("brightness +1:  ", sim(a, np.clip(a.astype(int) + 1, 0, 255).astype(np.uint8)))
```

```python {.marimo}

```