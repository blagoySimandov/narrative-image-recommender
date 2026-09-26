import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path

    return Path, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load the dataset
    Loading the dataset using a loader function
    """)
    return


@app.cell
def _(Path):
    from dataset_loaders import load_vist_album_images

    album_id = "72157608661271127"
    json_path = Path("datasets/sis/train.story-in-sequence.json")
    image_records = load_vist_album_images(
        album_id=album_id, json_path=json_path
    )
    return (image_records,)


@app.cell
def _(image_records, pl):
    image_df = pl.DataFrame(
        data=image_records,
    )
    return (image_df,)


@app.cell
def _(image_df, mo):
    mo.ui.table(
        data=image_df,
        format_mapping={"path": lambda path: mo.image(src=f"{path}")},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Getting access to the JEV
    Using openrouter SDK + OpenRouter key.
    From my research this type of model should not be that revolutionary and an opensource version of it (although probably worse in  accuracy) should be availble on github  and was realeased 2025.
    HackerNews article I read: https://news.ycombinator.com/item?id=49736660
    """)
    return


@app.cell
def _():
    # Source - https://stackoverflow.com/a/61029741
    # Posted by ParisNakitaKejser, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-09-20, License - CC BY-SA 4.0
    import os
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    return (os,)


@app.cell
def _(os):
    from openrouter import OpenRouter

    def pick_next_caption(
        current_caption: str, story_so_far: list, remaining: dict
    ) -> str:
        """remaining: {image_id: caption} for images not yet used in the story."""
        with OpenRouter(
            api_key=os.getenv("OPENROUTER_API_KEY", "")
        ) as open_router:
            res = open_router.alpha.decisions.create(
                model="typesafe/jev-1.13",
                state={
                    "current_caption": current_caption,
                    "story_so_far": story_so_far, #actual string story
                    "photos_remaining_and_not_used": len(remaining),
                },
                questions={
                    "next_image": {
                        "type": "choice",
                        "instructions": """
                        Given a selection of image captions and a current selected image caption your task is to pick the next image caption to create a story. You must try and advance the story always don't pick an image that is slightly different from the past one. Your story will be composed of only 5 images/captions so try to make it complete and intersting in only these 5 images.
                        """,
                        "criteria": remaining,  # {image_id: caption}
                    },
                },
            )
        return res.answers["next_image"].choice

    return (pick_next_caption,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Captioninig my images using BLIP from SalesForce
    """)
    return


@app.cell
def _(mo):
    radiogroup = mo.ui.radio(
        options=["large-blip", "base-blip"], label="Choose a captioning model"
    )
    radiogroup
    return (radiogroup,)


@app.cell
def _(radiogroup):
    decision = radiogroup.value
    return (decision,)


@app.cell
def _(decision, pl):
    from transformers import pipeline

    # -base or -large (both work) but -large works better
    BLIP_MODEL_NAME="Salesforce/blip-image-captioning-large" if decision == "large-blip" else "Salesforce/blip-image-captioning-base"

    def caption_images_df(
        image_df: pl.DataFrame,
        prompt: str = "", # absolutely useless. this shit just prepends the text to the caption!......
    ) -> pl.DataFrame:
        captioner = pipeline("image-text-to-text", model=BLIP_MODEL_NAME)
        captions = [
            captioner(images=str(p), text=prompt)[0]["generated_text"]
            for p in image_df["path"]
        ]
        return image_df.with_columns(pl.Series("caption", captions))

    return (caption_images_df,)


@app.cell
def _(caption_images_df, image_df):
    captioned_images_df = caption_images_df(image_df)
    return (captioned_images_df,)


@app.cell
def _(captioned_images_df, mo):
    table = mo.ui.table(captioned_images_df, format_mapping={"path": lambda path: mo.image(src=f"{path}")},
                selection="single"#using the selection as the "starter image" for the recommendation
               )
    table
    return (table,)


@app.cell
def _(captioned_images_df, pick_next_caption, pl, table):

    NUM_STEPS = 6

    selected_df = table.value
    current_id = selected_df["id"][0]
    current_caption = selected_df["caption"][0]

    remaining_df = captioned_images_df.filter(pl.col("id") != current_id)
    story_rows = [selected_df.row(0, named=True)] #init story_rows with selected row.

    for _ in range(min(NUM_STEPS - 1, remaining_df.height)):
        remaining = dict(zip(remaining_df["id"], remaining_df["caption"]))

        next_id = pick_next_caption(
            current_caption=current_caption,
            story_so_far=[r["caption"] for r in story_rows],
            remaining=remaining,
        )

        next_row = remaining_df.filter(pl.col("id") == next_id).row(0, named=True)
        story_rows.append(next_row)

        current_id = next_row["id"]
        current_caption = next_row["caption"]
        remaining_df = remaining_df.filter(pl.col("id") != next_id)
    return (story_rows,)


@app.cell
def _(mo, story_rows):
    #could import from image_pipeline
    def story_strip(mo, story_rows):
        def story_card(path, caption):
            return mo.vstack([
                mo.image(src=str(path), width=220),
                mo.md(f"**{caption}**"),
            ], align="center")

        cards = []
        for i, row in enumerate(story_rows):
            cards.append(story_card(row["path"], row["caption"]))
            if i < len(story_rows) - 1:
                cards.append(mo.md("### →"))

        return mo.hstack(cards, align="center", gap=1, wrap=True)

    story_strip(mo, story_rows)
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
