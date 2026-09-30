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
    # Coverage and Transition Selector

    This notebook selects $k$ images from an album and puts them in a sequence.
    The model uses CLIP image embeddings and cosine similarity $s(i, j)$.
    The objective has two terms.

    **Coverage.** Each image in the album must have one selected image that
    is similar to it. For a selected set $S$ of an album $A$:


    **==AI WAS HERE==**



    $$C(S) = \frac{1}{|A|} \sum_{i \in A} \max_{p \in S} s(i, p)$$

    Each selected image represents the album images that are nearest to it.
    A high $C(S)$ shows that the selection represents all of the album.

    **Transition.** Each selected image must be similar to the image before
    it and to the image after it.

    **==AI WAS HERE==**


    For a sequence $p_1, \dots, p_k$:

    $$T(S) = \frac{1}{k - 1} \max_{\text{order}} \sum_{t=1}^{k-1} s'(p_t, p_{t+1})$$




    The model tries all the sets and all the orders, and keeps the best.


    **I did not think of this idea about the duplicates AI suggested it without me even noticing a problem**

    A near-duplicate pair gives a very high transition score. To stop this,
    $s'$ is 0 when $s$ is equal to or more than the duplicate cap.

    **Objective.** $J(S) = \lambda \, C(S) + (1 - \lambda) \, T(S)$.
    The two terms have a conflict. Coverage prefers different images.
    Transition prefers similar images.
    """)
    return


@app.cell
def _():
    from itertools import batched, combinations, permutations
    from types import SimpleNamespace

    import numpy as np
    import polars as pl

    from dataset_loaders import load_cufed
    from image_pipeline import encode_images, load_clip, story_strip

    return (
        SimpleNamespace,
        batched,
        combinations,
        encode_images,
        load_clip,
        load_cufed,
        np,
        permutations,
        pl,
        story_strip,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Algorithm

    Aim is to maximize:

    J(S) = λ · C(S) + (1 − λ) · T(S)

    The search tries every set of $k$ images and every order of each set.
    It keeps the set and the order with the largest $J$. Thus the result is
    the optimal solution.

    Since with the current algorithm we do selection seperate from order we use lambda λ
    to denote the importance of each term of the equation.

    λ = 0.5 -> Both terms are equally important
    λ = 0 -> only transition matters
    λ = 1 -> only coverage matters
    """)
    return


@app.cell
def _(SimpleNamespace, batched, combinations, np, permutations):
    def best_in_batch(sim, transition_sim_matrix, sets, lam):
        k = sets.shape[1]
        # Permutations of N elements (max 20k) of class k
        perms = np.array([p for p in permutations(range(k)) if p[0] <= p[-1]]) # [0,23,1]

        #numpy magic since looping was slow as fuck
        orders = sets[:, perms] #numpy index via array index to get permutations of the set
        tr_all = transition_sim_matrix[orders[..., :-1], orders[..., 1:]].mean(axis=2) # 
        tr = tr_all.max(axis=1)
        cov = sim[:, sets].max(axis=2).mean(axis=0)
        j = lam * cov + (1 - lam) * tr
        b = int(j.argmax())




        return SimpleNamespace(
            indices=orders[b, tr_all[b].argmax()].tolist(),
            coverage=float(cov[b]),
            transition=float(tr[b]),
            objective=float(j[b]),
        )

    def select(sim, k, lam, cap=1.0, sets=None):
        """
        sim - nxn sim matrix
        k - number of images to select
        lam - lambda

        """
        transition_sim_matrix = np.where(sim >= cap, 0, sim) # filter ij pairs with similarity that is too big.
        sets = sets or combinations(range(len(sim)), k) # this includes 0
        batches = batched(sets, 20_000)  # from list of N makes K lists of 20_000 each - "batching"
        results = (best_in_batch(sim, transition_sim_matrix, np.array(b), lam) for b in batches)
        return max(results, key=lambda r: r.objective)

    return (select,)


@app.cell
def _(load_clip, load_cufed):
    cufed = load_cufed()
    processor, model, device = load_clip()
    return cufed, device, model, processor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set the Inputs

    Select a CUFED album. Then set $k$, $\lambda$, and the duplicate cap.
    """)
    return


@app.cell
def _(cufed, mo, pl):
    album_sizes = cufed.group_by("event_id").agg(
        pl.len().alias("images"), pl.col("event_types").first()
    )
    album_options = {
        f"{r['event_id']} ({', '.join(r['event_types'])}, {r['images']})": r["event_id"]
        for r in album_sizes.sort("event_id").iter_rows(named=True)
    }
    album_picker = mo.ui.dropdown(
        options=album_options, value=next(iter(album_options)), label="Album"
    )
    k_slider = mo.ui.slider(2, 5, value=4, label="k", debounce=True)
    lam_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="λ", debounce=True)
    cap_slider = mo.ui.slider(
        0.85, 1.0, step=0.01, value=0.95, label="Duplicate cap", debounce=True
    )
    mo.vstack([album_picker, mo.hstack([k_slider, lam_slider, cap_slider])])
    return album_picker, cap_slider, k_slider, lam_slider


@app.cell
def _(album_picker, cufed, device, encode_images, model, pl, processor):
    album = cufed.filter(pl.col("event_id") == album_picker.value)
    embeddings = encode_images(album["path"].to_list(), processor, model, device)

    # cosine similary via matrix multiplication of unit vectors
    # this works coz clip uses unit vectors when encoding stuff
    # Gives us nxn similarity matrix where i,j is a similarity between image i and j
    # example https://github.com/openai/CLIP#zero-shot-prediction
    # note that openai are normalizing them in this example but everywhere i check it says they are automatically
    # l2 normalized. I have no idea why they are normalizing them before doing the calculation here.
    sim = (embeddings @ embeddings.T).cpu().numpy()
    return album, sim


@app.cell
def _(sim):
    sim.diagonal() #check that they are actually normalized
    return


@app.cell
def _(cap_slider, k_slider, lam_slider, select, sim):
    selection = select(sim, k_slider.value, lam_slider.value, cap_slider.value)
    return (selection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selected Sequence

    The strip shows the selected images in the best order. The value between
    two images is their cosine similarity.
    """)
    return


@app.cell
def _(album, mo, selection, sim, story_strip):
    _paths = album["path"].to_list()
    _order = selection.indices
    _rows = [
        {
            "path": _paths[p],
            "caption": f"{t + 1} · next {sim[p, _order[t + 1]]:.2f}"
            if t < len(_order) - 1
            else f"{t + 1}",
        }
        for t, p in enumerate(_order)
    ]
    mo.vstack(
        [
            mo.md(
                f"Coverage **{selection.coverage:.3f}** · "
                f"Transition **{selection.transition:.3f}** · "
                f"Objective **{selection.objective:.3f}**"
            ),
            story_strip(mo, _rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iconic Images

    Group the selected images from the album with the ones that were note selected to find the the clusters
    """)
    return


@app.cell
def _(np):
    def assign_to_picks(sim: np.ndarray, picks: list[int]) -> np.ndarray:
        return np.asarray(picks)[sim[:, picks].argmax(axis=1)] 



    return (assign_to_picks,)


@app.cell
def _(mo):
    def image_grid(paths: list[str], width: int = 110):
        cards = [mo.image(src=p, width=width) for p in paths]
        return mo.hstack(cards, wrap=True, justify="start", gap=0.25)

    return (image_grid,)


@app.cell
def _(album, assign_to_picks, image_grid, mo, selection, sim):
    _paths = album["path"].to_list()
    _owner = assign_to_picks(sim, selection.indices)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.image(src=_paths[p], width=160),
                            mo.md(f"<small>{int((_owner == p).sum())} images</small>"),
                        ]
                    ),
                    image_grid(
                        [_paths[i] for i in (_owner == p).nonzero()[0] if i != p]
                    ),
                ],
                justify="start",
                align="start",
                widths=[1, 5],
            )
            for p in selection.indices
        ],
        gap=1,
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
