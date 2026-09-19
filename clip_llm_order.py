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
    from dataclasses import dataclass
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    import matplotlib.pyplot as plt
    from PIL import Image

    from clip_utils import (
        encode_images,
        encode_text,
        find_album_images,
        load_clip,
        rank_images,
    )

    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BLIP_MODEL_NAME,
        Image,
        Path,
        dataclass,
        encode_images,
        encode_text,
        find_album_images,
        load_clip,
        pipeline,
        plt,
        rank_images,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set the Inputs
    """)
    return


@app.cell
def _(Path):
    json_path = Path("sis/test.story-in-sequence.json")
    album_id = "504823"
    top_k = 5

    NUM_STEPS = 5
    TEMPERATURE = 1.0
    MAX_NEW_TOKENS = 30
    MAX_SENTENCE_WORDS = 15
    return MAX_NEW_TOKENS, MAX_SENTENCE_WORDS, NUM_STEPS, TEMPERATURE, album_id


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
def _(album_id, find_album_images):
    image_paths = find_album_images(album_id)
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
def _(BLIP_MODEL_NAME, Path, dataclass, encode_images, pipeline):
    @dataclass
    class ImageEntry:
        path: Path
        embedding: object
        caption: str


    class ImageBase:
        def __init__(self, entries: list):
            self.entries = entries

        def __len__(self) -> int:
            return len(self.entries)

        def __iter__(self):
            return iter(self.entries)
    
        def __getitem__(self, index: int):
            return self.entries[index]

        @property
        def paths(self) -> list:
            return [e.path for e in self.entries]

        @property
        def embeddings(self):
            import torch

            return torch.stack([e.embedding for e in self.entries])

        @property
        def captions(self) -> dict:
            return {e.path: e.caption for e in self.entries}

        def remove(self, path: Path) -> None:
            self.entries = [e for e in self.entries if e.path != path]


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a Next-Sentence Prompt

    This function grounds the LLM in the captions of the images still
    available, and asks for one short sentence describing what happens next.
    """)
    return


@app.function
def build_next_step_prompt(current_caption: str, other_captions: list, max_words: int) -> str:
    captions_text = "\n".join(f"- {c}" for c in other_captions)
    return (
        f"Here are other photos in this album:\n{captions_text}\n\n"
        f"The current photo shows: {current_caption}\n\n"
        "Write one short sentence describing what happens NEXT in the story, "
        "after this photo. Do not describe the current photo. "
        "Only write about things that could plausibly appear in one of the "
        f"other photos listed.\nWrite exactly one sentence, no more than "
        f"{max_words} words."
    )


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
    NUM_STEPS,
    TEMPERATURE,
    client,
    device,
    encode_images,
    encode_text,
    image_base,
    model,
    processor,
    rank_images,
):
    remaining = list(image_base)
    current_entry = remaining.pop(0)
    story = []
    story_images = [current_entry.path]
    for _ in range(min(NUM_STEPS, len(remaining) + 1)):
        other_captions = [e.caption for e in remaining]
        prompt = build_next_step_prompt(current_entry.caption, other_captions, MAX_SENTENCE_WORDS)
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
        current_entry = next((e for e in remaining if e.path == best_path))
        remaining = [e for e in remaining if e.path != best_path]
    return story, story_images


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the Story

    This shows the images picked by the loop above, in the order they were
    chosen, with the generated sentence for each one.
    """)
    return


@app.cell
def _(Image, plt, story, story_images):
    fig, axes = plt.subplots(1, len(story_images), figsize=(4 * len(story_images), 4))
    if len(story_images) == 1:
        axes = [axes]
    for ax, path, _sentence in zip(axes, story_images, story):
        ax.imshow(Image.open(path))
        ax.set_title(_sentence, fontsize=9, wrap=True)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
