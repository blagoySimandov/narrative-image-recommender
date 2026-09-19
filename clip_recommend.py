import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CLIP Image Recommender

    This notebook takes a text query. It finds the images in an album that
    match the query.

    CLIP has two neural networks: an image encoder and a text encoder. The two
    networks train together. This training puts the output of each network in
    the same vector space.

    CLIP  is trained on 400
    million pairs of an image and a caption from the internet. The training
    task is "Does this image match this caption, or not?"
    Because of this method, CLIP can work with any text query at run time.

    Reference: CLIP paper, https://arxiv.org/abs/2103.00020
    """)
    return


@app.cell
def _():
    from pathlib import Path

    from clip_utils import (
        encode_images,
        encode_text,
        find_album_images,
        load_clip,
        plot_results_grid,
        rank_images,
    )

    return (
        Path,
        encode_images,
        encode_text,
        find_album_images,
        load_clip,
        plot_results_grid,
        rank_images,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set the Inputs

    Set the json path, the album id, and the top k value here. Change these
    values to try a different album.
    """)
    return


@app.cell
def _(Path):
    json_path = Path("datasets/sis/test.story-in-sequence.json")
    album_id = "504823"
    top_k = 5
    return album_id, top_k


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the CLIP Model

    This step loads the CLIP model and the processor. This step can take a
    moment on the first run, since it downloads the model weights.
    """)
    return


@app.cell
def _(load_clip):
    processor, model, device = load_clip()
    return device, model, processor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find the Images

    This step finds the downloaded images for the album set above.
    """)
    return


@app.cell
def _(album_id, find_album_images):
    image_paths = find_album_images(album_id)
    image_paths
    return (image_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the Encoders

    This step encodes the images with CLIP.
    """)
    return


@app.cell
def _(device, encode_images, image_paths, model, processor):
    image_features = encode_images(image_paths, processor, model, device)
    return (image_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare Many Queries

    This step tests a list of queries against the same album. Edit the
    `queries` list to test different text.
    """)
    return


@app.cell
def _(
    device,
    encode_text,
    image_features,
    image_paths,
    model,
    plot_results_grid,
    processor,
    rank_images,
    top_k,
):
    queries = [
        "cat going to sleep",
        "people posing for a photo",
        "playing a game",
    ]

    results_by_query = {}
    for q in queries:
        text_feature = encode_text(q, processor, model, device)
        results_by_query[q] = rank_images(
            image_paths, image_features, text_feature, top_k
        )

    plot_results_grid(results_by_query, top_k)
    return


if __name__ == "__main__":
    app.run()
