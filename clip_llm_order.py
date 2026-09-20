import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    ## type: ignore
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CLIP LLM Ordering

    The idea is to take the pipeline in `clip_recommend` and add a text
    generation step using an LLM asking the question "How would you complete
    this sentence?"

    Given this request, the model completes the sentence. With the completed
    sentence, we run text embedding again and search for the closest image in
    the dataset that matches this text.

    We could rerun the text generation a few times with a high temperature, to
    get different text completions. This would help build "branches" of
    stories, and show how the same images can be ordered in different ways.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    import matplotlib.pyplot as plt
    from PIL import Image

    from image_pipeline import (
        ImageBase,
        ImageEntry,
        encode_images,
        encode_text,
        load_clip,
        rank_images,
    )
    from dataset_loaders import (
        load_personal_photos_images,
        load_vist_album_images,
        load_yfcc_album_images,
    )

    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BLIP_MODEL_NAME,
        ImageBase,
        ImageEntry,
        Path,
        encode_images,
        encode_text,
        load_clip,
        load_personal_photos_images,
        load_vist_album_images,
        load_yfcc_album_images,
        pipeline,
        rank_images,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set Configuration

    Pick which dataset to run the pipeline against.
    """)
    return


@app.cell
def _(mo):
    dataset_choice = mo.ui.radio(
        options=["yfcc_amsterdam", "vist", "personal_photos"],
        value="yfcc_amsterdam",
        label="Dataset",
    )
    dataset_choice
    return (dataset_choice,)


@app.cell
def _(Path, dataset_choice):
    max_photos = 50
    top_k = 5

    NUM_STEPS = 5
    TEMPERATURE = 1.0
    MAX_NEW_TOKENS = 30
    MAX_SENTENCE_WORDS = 15

    city = None
    city_dir = None
    album_id = None
    json_path = None

    if dataset_choice.value == "yfcc_amsterdam":
        city = "Amsterdam"
        city_dir = Path("datasets/yfcmmf00m-cities-amsterdam")
        album_id = "98"
    elif dataset_choice.value == "vist":
        album_id = "72157608661271127"
        json_path = Path("datasets/sis/train.story-in-sequence.json")
    return (
        MAX_NEW_TOKENS,
        MAX_SENTENCE_WORDS,
        NUM_STEPS,
        TEMPERATURE,
        album_id,
        city,
        city_dir,
        json_path,
        max_photos,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the CLIP Model
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
    """)
    return


@app.cell
def _(
    album_id,
    city,
    city_dir,
    dataset_choice,
    json_path,
    load_personal_photos_images,
    load_vist_album_images,
    load_yfcc_album_images,
    max_photos,
):
    if dataset_choice.value == "yfcc_amsterdam":
        image_records = load_yfcc_album_images(city, album_id, city_dir)[:max_photos]
    elif dataset_choice.value == "vist":
        image_records = load_vist_album_images(album_id, json_path)[:max_photos]
    else:
        image_records = load_personal_photos_images()[:max_photos]
    image_paths = [r.path for r in image_records]
    return (image_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Image Base

    `ImageEntry` stores one image's CLIP embedding and BLIP caption together.
    `ImageBase` holds all the entries for an album, with easy access to the
    embeddings as one tensor and the captions as one dict.
    """)
    return


@app.cell
def _(BLIP_MODEL_NAME, ImageBase, ImageEntry, encode_images, pipeline):
    def load_captioner():
        return pipeline("image-text-to-text", model=BLIP_MODEL_NAME)


    def build_image_base(image_paths: list, processor, model, device: str, captioner) -> ImageBase:
        embeddings = encode_images(image_paths, processor, model, device)
        entries = []
        for path, embedding in zip(image_paths, embeddings):
            caption = captioner(images=str(path),text="")[0]["generated_text"]
            entries.append(ImageEntry(path=path, embedding=embedding, caption=caption))
        return ImageBase(entries)

    return build_image_base, load_captioner


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the Image Base

    This encodes every image with CLIP and captions every image with BLIP.
    """)
    return


@app.cell
def _(build_image_base, device, image_paths, load_captioner, model, processor):
    captioner = load_captioner()
    image_base = build_image_base(image_paths, processor, model, device, captioner)

    for entry in image_base:
        print(f"{entry.path.name}: {entry.caption}")
    return (image_base,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The LLM Client

    `LLMClient` loads a small local language model and generates text from a
    prompt. It keeps the model, tokenizer, and device together on one object.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, torch):
    class LLMClient:
        def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
            self.model.eval()

        def generate(self, prompt: str, temperature: float = 1.0, max_new_tokens: int = 40) -> str:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,

                    #Stop the fucker from repeating the same stuff
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return (LLMClient,)


@app.cell
def _(LLMClient):
    client = LLMClient()
    return (client,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a Next-Sentence Prompt

    This function grounds the LLM in the captions of the images still
    available, and asks for one short sentence describing what happens next.
    """)
    return


@app.cell
def _():
    story = []
    NEXT_STEP_PROMPT_TEMPLATE = """
    Your task is to complete the next sentance of a story. Each time I will be giving you the current story +
    a selection of image captions. The idea is to build a story from the images. Your task is just to generate the next sentance
    DO NOT PUT ANYHTING ELSE IN THE RESPONSE EXCEPT THE NEXT STORY SENTANCE.
    Be original. If the first story shows "A" don't just assume "A" will continue - you need to advance the story forward and not repeat it. If two people are rigind bikes don't continue the story with them riding bikes or if you do add something interesting.


    Here are other photos in this album:
    {captions_text}

    The current photo shows: {current_caption}

    The story up to this point is:
    {story}

    Write exactly one sentence, no more than {max_words} words."""


    # Idea of v2 is to not pollute the context with the whole story but instead only provide the last sentence. This is to avoid the model repeating itself and to keep the story moving forward.
    NEXT_STEP_PROMPT_TEMPLATE_V2 = """You are writing a photo story, one sentence at a time.

    Last sentence of the story: {last_sentence}

    Other photo captions available in this album:
    {captions_text}

    The current photo shows: {current_caption}

    Write ONE new sentence continuing the story. Rules:
    - Do not repeat any words or phrasing from the last sentence.
    - Do not describe the current photo again.
    - Move the story to a new action, place, or detail — not more of the same thing.
    - Only write about things that could plausibly appear in one of the other photos listed.
    - Output ONLY the sentence. No preamble, no quotes.

    Maximum {max_words} words."""

    def build_next_step_prompt(current_caption: str, other_captions: list, max_words: int, prompt:str) -> str:
        captions_text = "\n".join(f"- {c}" for c in other_captions)
        return prompt.format(
            captions_text=captions_text,
            current_caption=current_caption,
            last_sentence=story[-1] if len(story) >0 else "(this is the first sentence)",
            max_words=max_words,
        )

    return NEXT_STEP_PROMPT_TEMPLATE_V2, build_next_step_prompt, story


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the Story Loop

    Each step asks the LLM for the next sentence, grounded in the captions of
    the images not yet used. It embeds that sentence with CLIP and picks the
    closest remaining image. That image is removed from the pool, and the loop
    continues for `NUM_STEPS` steps.
    """)
    return


@app.cell
def _(
    MAX_NEW_TOKENS,
    MAX_SENTENCE_WORDS,
    NEXT_STEP_PROMPT_TEMPLATE_V2,
    NUM_STEPS,
    TEMPERATURE,
    build_next_step_prompt,
    client,
    device,
    encode_images,
    encode_text,
    image_base,
    model,
    processor,
    rank_images,
    story,
):
    remaining = list(image_base)
    current_entry = remaining.pop(0)
    story_images = [current_entry.path]
    def run_story_loop(prompt_template: str, remaining: list, current_entry, story_images: list, story: list):
        for _ in range(min(NUM_STEPS, len(remaining) + 1)):
            other_captions = [e.caption for e in remaining]
            prompt = build_next_step_prompt(current_entry.caption, other_captions, MAX_SENTENCE_WORDS, prompt_template)
            _sentence = client.generate(prompt, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
            story.append(_sentence)
            if not remaining:
                break
            remaining_paths = [e.path for e in remaining]
            remaining_features = encode_images(remaining_paths, processor, model, device)
            text_feature = encode_text(_sentence, processor, model, device)
            best_path, score = rank_images(remaining_paths, remaining_features, text_feature, top_k=1)[0]
            print(f'{_sentence}  ->  {best_path.name} (score={score:.4f})')
            story_images.append(best_path)
            current_entry = next(e for e in remaining if e.path == best_path)
            remaining = [e for e in remaining if e.path != best_path]
        return story_images, story

    story_images, _=run_story_loop(NEXT_STEP_PROMPT_TEMPLATE_V2, remaining, current_entry, story_images, story)
    return (story_images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the Story

    This shows the images picked by the loop above, in the order they were
    chosen, with the generated sentence for each one.
    """)
    return


@app.cell
def _(image_base, mo, story, story_images):
    def story_card(path, caption, sentence):
        return mo.vstack([
            mo.image(src=str(path), width=220),
            mo.md(f"**Caption:** {caption}"),
            mo.md(f"**LLM:** {sentence}"),
        ], align="center")

    cards = []
    for i, (path, sentence) in enumerate(zip(story_images, story)):
        caption = image_base.captions.get(path, "")
        cards.append(story_card(path, caption, sentence))
        if i < len(story_images) - 1:
            cards.append(mo.md("### ->"))

    mo.hstack(cards, align="center", gap=1, wrap=True)
    return


@app.cell
def _(image_base):
    caption_list = [i.caption for i in image_base]
    caption_list
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
