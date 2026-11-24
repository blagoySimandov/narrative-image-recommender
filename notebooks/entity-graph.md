---
title: Entity Graph
marimo-version: 0.24.2
width: full
---

```python {.marimo}
import marimo as mo
import polars as pl
import numpy as np
from dataset_loaders import load_yfcc_city
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from networkx import DiGraph
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)
import torch
```

Load the amsterdam dataset

```python {.marimo}
amsterdam = load_yfcc_city("amsterdam")
amsterdam
```

## Load an OpenVocab/Labeled Segmentation Model

Load the vocabolary

```python {.marimo}
lvis_vocab: list[str] = ['Band Aid','CD player',    'Christmas tree',    'Dixie cup',    'Ferris wheel',    'French toast',    'Lego',    'aerosol can',    'air conditioner',    'airplane',    'alarm clock',    'alcohol',    'alligator',    'almond',    'ambulance',    'amplifier',    'anklet',    'antenna',    'apple',    'apron',    'aquarium',    'armband',    'armchair',    'artichoke',    'ashtray',    'asparagus',    'atomizer',    'automatic washer',    'avocado',    'award',    'awning',    'baby buggy',    'backpack',    'bagel',    'ball',    'balloon',    'bamboo',    'banana',    'bandage',    'bandanna',
    'banner',
    'barrel',
    'barrette',
    'barrow',
    'baseball',
    'baseball base',
    'baseball bat',
    'baseball cap',
    'baseball glove',
    'basket',
    'basketball',
    'basketball backboard',
    'bat',
    'bath mat',
    'bath towel',
    'bathrobe',
    'bathtub',
    'battery',
    'bead',
    'bean curd',
    'beanbag',
    'beanie',
    'bear',
    'bed',
    'bedspread',
    'beef',
    'beer bottle',
    'beer can',
    'bell',
    'bell pepper',
    'belt',
    'belt buckle',
    'bench',
    'beret',
    'bib',
    'bicycle',
    'billboard',
    'binder',
    'binoculars',
    'bird',
    'birdbath',
    'birdcage',
    'birdfeeder',
    'birdhouse',
    'birthday cake',
    'black sheep',
    'blackberry',
    'blackboard',
    'blanket',
    'blazer',
    'blender',
    'blinder',
    'blinker',
    'blouse',
    'blueberry',
    'boat',
    'bobbin',
    'bobby pin',
    'boiled egg',
    'bolt',
    'book',
    'bookcase',
    'booklet',
    'boot',
    'bottle',
    'bottle cap',
    'bottle opener',
    'bouquet',
    'bow',
    'bow-tie',
    'bowl',
    'bowler hat',
    'box',
    'bracelet',
    'brake light',
    'brassiere',
    'bread',
    'bread-bin',
    'bridal gown',
    'briefcase',
    'broccoli',
    'broom',
    'brownie',
    'brussels sprouts',
    'bucket',
    'bull',
    'bulldog',
    'bullet train',
    'bulletin board',
    'bullhorn',
    'bun',
    'bunk bed',
    'buoy',
    'bus',
    'business card',
    'butter',
    'butterfly',
    'button',
    'cab',
    'cabin car',
    'cabinet',
    'cake',
    'calculator',
    'calendar',
    'calf',
    'camcorder',
    'camel',
    'camera',
    'camera lens',
    'camper',
    'can',
    'can opener',
    'candle',
    'candle holder',
    'candy cane',
    'canister',
    'canoe',
    'cantaloup',
    'cap',
    'cape',
    'cappuccino',
    'car',
    'card',
    'cardigan',
    'carrot',
    'cart',
    'carton',
    'cash register',
    'cast',
    'cat',
    'cauliflower',
    'cayenne',
    'celery',
    'cellular telephone',
    'chair',
    'chandelier',
    'cherry',
    'chicken',
    'chickpea',
    'chili',
    'chocolate bar',
    'chocolate cake',
    'choker',
    'chopping board',
    'chopstick',
    'cigarette',
    'cigarette case',
    'cincture',
    'cistern',
    'clasp',
    'cleansing agent',
    'clip',
    'clipboard',
    'clock',
    'clock tower',
    'clothes hamper',
    'clothespin',
    'coaster',
    'coat',
    'coat hanger',
    'coatrack',
    'cock',
    'coconut',
    'coffee maker',
    'coffee table',
    'coffeepot',
    'coin',
    'colander',
    'coleslaw',
    'computer keyboard',
    'condiment',
    'cone',
    'control',
    'cookie',
    'cooler',
    'cork',
    'corkscrew',
    'cornet',
    'cornice',
    'corset',
    'costume',
    'cover',
    'cow',
    'cowbell',
    'cowboy hat',
    'crab',
    'cracker',
    'crate',
    'crayon',
    'crescent roll',
    'crib',
    'crisp',
    'crock pot',
    'crossbar',
    'crow',
    'crown',
    'crucifix',
    'cruise ship',
    'crumb',
    'crutch',
    'cub',
    'cube',
    'cucumber',
    'cufflink',
    'cup',
    'cupboard',
    'cupcake',
    'curtain',
    'cushion',
    'dartboard',
    'deadbolt',
    'deck chair',
    'deer',
    'dental floss',
    'desk',
    'diaper',
    'dining table',
    'dirt bike',
    'dish',
    'dish antenna',
    'dishrag',
    'dishtowel',
    'dishwasher',
    'dispenser',
    'dog',
    'dog collar',
    'doll',
    'dolphin',
    'domestic ass',
    'doorknob',
    'doormat',
    'doughnut',
    'drawer',
    'dress',
    'dress hat',
    'dress suit',
    'dresser',
    'drill',
    'drum',
    'duck',
    'duckling',
    'duct tape',
    'duffel bag',
    'dumpster',
    'eagle',
    'earphone',
    'earring',
    'easel',
    'edible corn',
    'egg',
    'egg yolk',
    'eggbeater',
    'eggplant',
    'elephant',
    'elk',
    'envelope',
    'eraser',
    'fan',
    'faucet',
    'ferry',
    'fighter jet',
    'figurine',
    'file cabinet',
    'fire alarm',
    'fire engine',
    'fire extinguisher',
    'fire hose',
    'fireplace',
    'fireplug',
    'fish',
    'fishing rod',
    'flag',
    'flagpole',
    'flamingo',
    'flannel',
    'flap',
    'flashlight',
    'flip-flop',
    'flipper',
    'flower arrangement',
    'flowerpot',
    'flute glass',
    'foal',
    'folding chair',
    'food processor',
    'football',
    'footstool',
    'fork',
    'forklift',
    'freight car',
    'freshener',
    'frisbee',
    'frog',
    'fruit juice',
    'frying pan',
    'fume hood',
    'garbage truck',
    'garden hose',
    'gargle',
    'garlic',
    'gazelle',
    'gelatin',
    'giant panda',
    'gift wrap',
    'ginger',
    'giraffe',
    'glass',
    'globe',
    'glove',
    'goat',
    'goggles',
    'golf club',
    'golfcart',
    'goose',
    'grape',
    'grater',
    'gravestone',
    'green bean',
    'green onion',
    'grill',
    'grizzly',
    'grocery bag',
    'guitar',
    'gull',
    'gun',
    'hair dryer',
    'hairbrush',
    'hairnet',
    'hairpin',
    'ham',
    'hamburger',
    'hammer',
    'hammock',
    'hamster',
    'hand towel',
    'handbag',
    'handcart',
    'handkerchief',
    'handle',
    'hat',
    'headband',
    'headboard',
    'headlight',
    'headscarf',
    'headstall',
    'heart',
    'heater',
    'helicopter',
    'helmet',
    'highchair',
    'hinge',
    'hog',
    'home plate',
    'honey',
    'hook',
    'horse',
    'horse carriage',
    'hose',
    'hot sauce',
    'hummingbird',
    'iPod',
    'ice maker',
    'icecream',
    'identity card',
    'igniter',
    'iron',
    'ironing board',
    'jacket',
    'jam',
    'jar',
    'jean',
    'jeep',
    'jersey',
    'jet plane',
    'jewelry',
    'jumpsuit',
    'kayak',
    'kettle',
    'key',
    'kilt',
    'kimono',
    'kitchen sink',
    'kite',
    'kitten',
    'kiwi fruit',
    'knee pad',
    'knife',
    'knob',
    'ladder',
    'ladle',
    'ladybug',
    'lamb',
    'lamp',
    'lamppost',
    'lampshade',
    'lantern',
    'lanyard',
    'laptop computer',
    'latch',
    'legging',
    'lemon',
    'lettuce',
    'license plate',
    'life buoy',
    'life jacket',
    'lightbulb',
    'lime',
    'lion',
    'lip balm',
    'lizard',
    'log',
    'lollipop',
    'loveseat',
    'magazine',
    'magnet',
    'mail slot',
    'mailbox',
    'mandarin orange',
    'manger',
    'manhole',
    'map',
    'marker',
    'mashed potato',
    'mask',
    'mast',
    'mat',
    'mattress',
    'measuring cup',
    'measuring stick',
    'meatball',
    'medicine',
    'melon',
    'microphone',
    'microwave oven',
    'milk',
    'minivan',
    'mirror',
    'mitten',
    'mixer',
    'money',
    'monitor computer monitor',
    'monkey',
    'mop',
    'motor',
    'motor scooter',
    'motorcycle',
    'mound',
    'mouse',
    'mousepad',
    'muffin',
    'mug',
    'mushroom',
    'musical instrument',
    'napkin',
    'necklace',
    'necktie',
    'needle',
    'nest',
    'newspaper',
    'newsstand',
    'nightshirt',
    'noseband',
    'notebook',
    'notepad',
    'nut',
    'oar',
    'oil lamp',
    'olive oil',
    'onion',
    'orange',
    'orange juice',
    'ostrich',
    'ottoman',
    'oven',
    'overalls',
    'owl',
    'pacifier',
    'packet',
    'pad',
    'paddle',
    'padlock',
    'paintbrush',
    'painting',
    'pajamas',
    'palette',
    'pan',
    'pancake',
    'paper plate',
    'paper towel',
    'parachute',
    'parakeet',
    'parasail',
    'parasol',
    'parka',
    'parking meter',
    'parrot',
    'passenger car',
    'passport',
    'pastry',
    'pea',
    'peach',
    'peanut butter',
    'pear',
    'peeler',
    'pelican',
    'pen',
    'pencil',
    'penguin',
    'pepper',
    'pepper mill',
    'perfume',
    'person',
    'pet',
    'pew',
    'phonograph record',
    'piano',
    'pickle',
    'pickup truck',
    'pie',
    'pigeon',
    'pillow',
    'pineapple',
    'pinecone',
    'pipe',
    'pita',
    'pitcher',
    'pizza',
    'place mat',
    'plastic bag',
    'plate',
    'platter',
    'pliers',
    'pocketknife',
    'poker',
    'polar bear',
    'pole',
    'police cruiser',
    'polo shirt',
    'pony',
    'pop',
    'postbox',
    'postcard',
    'poster',
    'pot',
    'potato',
    'potholder',
    'pottery',
    'pouch',
    'power shovel',
    'prawn',
    'pretzel',
    'printer',
    'projectile',
    'projector',
    'propeller',
    'pumpkin',
    'puppy',
    'quilt',
    'rabbit',
    'racket',
    'radiator',
    'radio receiver',
    'radish',
    'raft',
    'railcar',
    'raincoat',
    'ram',
    'raspberry',
    'razorblade',
    'reamer',
    'rearview mirror',
    'receipt',
    'recliner',
    'record player',
    'reflector',
    'refrigerator',
    'remote control',
    'rhinoceros',
    'rifle',
    'ring',
    'robe',
    'rocking chair',
    'rolling pin',
    'router',
    'rubber band',
    'runner',
    'saddle',
    'saddle blanket',
    'saddlebag',
    'sail',
    'salad',
    'salami',
    'salmon',
    'salsa',
    'saltshaker',
    'sandal',
    'sandwich',
    'saucer',
    'sausage',
    'scale',
    'scarf',
    'school bus',
    'scissors',
    'scoreboard',
    'screwdriver',
    'scrubbing brush',
    'sculpture',
    'seabird',
    'seahorse',
    'seashell',
    'sewing machine',
    'shaker',
    'shampoo',
    'shark',
    'shaving cream',
    'sheep',
    'shield',
    'shirt',
    'shoe',
    'shopping bag',
    'shopping cart',
    'short pants',
    'shoulder bag',
    'shovel',
    'shower curtain',
    'shower head',
    'signboard',
    'silo',
    'sink',
    'skateboard',
    'skewer',
    'ski',
    'ski boot',
    'ski parka',
    'ski pole',
    'skirt',
    'sled',
    'sleeping bag',
    'slide',
    'slipper',
    'snowboard',
    'snowman',
    'snowmobile',
    'soap',
    'soccer ball',
    'sock',
    'sofa',
    'solar array',
    'soup',
    'soupspoon',
    'sour cream',
    'spatula',
    'speaker',
    'spectacles',
    'spice rack',
    'spider',
    'sponge',
    'spoon',
    'sportswear',
    'spotlight',
    'squirrel',
    'stapler',
    'starfish',
    'statue',
    'steak',
    'steering wheel',
    'step stool',
    'stereo',
    'stirrup',
    'stool',
    'stop sign',
    'stove',
    'strainer',
    'strap',
    'straw',
    'strawberry',
    'street sign',
    'streetlight',
    'suit',
    'suitcase',
    'sunflower',
    'sunglasses',
    'sunhat',
    'surfboard',
    'sushi',
    'suspenders',
    'sweat pants',
    'sweatband',
    'sweater',
    'sweatshirt',
    'sweet potato',
    'swimsuit',
    'sword',
    'table',
    'table lamp',
    'tablecloth',
    'tag',
    'taillight',
    'tank',
    'tank top',
    'tape',
    'tape measure',
    'tapestry',
    'tarp',
    'tartan',
    'tassel',
    'tea bag',
    'teacup',
    'teakettle',
    'teapot',
    'teddy bear',
    'telephone',
    'telephone booth',
    'telephone pole',
    'television camera',
    'television set',
    'tennis ball',
    'tennis racket',
    'thermometer',
    'thermos bottle',
    'thermostat',
    'thread',
    'thumbtack',
    'tiara',
    'tiger',
    'tights',
    'timer',
    'tinfoil',
    'tinsel',
    'tissue paper',
    'toast',
    'toaster',
    'toaster oven',
    'toilet',
    'toilet tissue',
    'tomato',
    'tongs',
    'toolbox',
    'toothbrush',
    'toothpaste',
    'toothpick',
    'tortilla',
    'tote bag',
    'tow truck',
    'towel',
    'towel rack',
    'toy',
    'tractor',
    'traffic light',
    'trailer truck',
    'train',
    'trash can',
    'tray',
    'tricycle',
    'tripod',
    'trophy cup',
    'trousers',
    'truck',
    'trunk',
    'turban',
    'turkey',
    'turtle',
    'turtleneck',
    'typewriter',
    'umbrella',
    'underdrawers',
    'underwear',
    'urinal',
    'urn',
    'vacuum cleaner',
    'vase',
    'veil',
    'vending machine',
    'vent',
    'vest',
    'videotape',
    'visor',
    'volleyball',
    'waffle',
    'wagon',
    'wagon wheel',
    'walking cane',
    'walking stick',
    'wall clock',
    'wall socket',
    'wallet',
    'watch',
    'water bottle',
    'water cooler',
    'water faucet',
    'water jug',
    'water scooter',
    'water ski',
    'water tower',
    'watering can',
    'watermelon',
    'weathervane',
    'webcam',
    'wedding cake',
    'wedding ring',
    'wet suit',
    'wheel',
    'wheelchair',
    'whipped cream',
    'whistle',
    'wig',
    'wind chime',
    'windmill',
    'window box',
    'windshield wiper',
    'windsock',
    'wine bottle',
    'wine bucket',
    'wineglass',
    'wok',
    'wooden spoon',
    'wreath',
    'wrench',
    'wristband',
    'wristlet',
    'yacht',
    'yogurt',
    'yoke',
    'zebra',
    'zucchini',
]
```

## Load the images

```python {.marimo}
from PIL import Image
images_loaded: list[Image.Image] = [Image.open(p) for p in amsterdam["path"]]
```

## Use SigLIP to label each image with objects and scenes

SigLIP scores each image against a list of text labels.
The object labels come from LVIS. The scene labels come from Places365.

```python {.marimo}
import urllib.request
from transformers import AutoModel, AutoProcessor

device = "cpu"
siglip_checkpoint = "google/siglip-so400m-patch14-384"
siglip_processor = AutoProcessor.from_pretrained(siglip_checkpoint)
siglip_model = AutoModel.from_pretrained(siglip_checkpoint).to(device).eval()  # pyright: ignore[reportArgumentType]
siglip_batch_size = 8
siglip_top_k = {"object": 10, "scene": 3}
```

```python {.marimo}
PLACES365_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"


def parse_place(line: str) -> str:
    return line.split()[0][3:].replace("/", " ").replace("_", " ")


def places365_vocab() -> list[str]:
    text = urllib.request.urlopen(PLACES365_URL).read().decode()
    return [parse_place(line) for line in text.splitlines() if line.strip()]


def build_vocab() -> pl.DataFrame:
    scenes = places365_vocab()
    return pl.DataFrame(
        {
            "label": lvis_vocab + scenes,
            "kind": ["object"] * len(lvis_vocab) + ["scene"] * len(scenes),
        }
    )


vocab = build_vocab()
vocab.group_by("kind").len()
```

```python {.marimo}
def as_tensor(features) -> torch.Tensor:
    return features if isinstance(features, torch.Tensor) else features.pooler_output


def embed_text(labels: list[str], chunk: int = 128) -> torch.Tensor:
    out = []
    for start in range(0, len(labels), chunk):
        prompts = [f"a photo of {l}" for l in labels[start : start + chunk]]
        inputs = siglip_processor(
            text=prompts, padding="max_length", max_length=64, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out.append(as_tensor(siglip_model.get_text_features(**inputs)))
    return torch.nn.functional.normalize(torch.cat(out), dim=-1)


def embed_images(images: list[Image.Image]) -> torch.Tensor:
    out = []
    for start in range(0, len(images), siglip_batch_size):
        batch = images[start : start + siglip_batch_size]
        inputs = siglip_processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(as_tensor(siglip_model.get_image_features(**inputs)))
    return torch.nn.functional.normalize(torch.cat(out), dim=-1)


text_emb = embed_text(vocab["label"].to_list())
image_emb = embed_images(images_loaded)
```

```python {.marimo}
def siglip_probabilities(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    scale = siglip_model.logit_scale.exp()
    return torch.sigmoid(img @ txt.T * scale + siglip_model.logit_bias).detach()


def top_k_for_kind(probs: torch.Tensor, kind: str, k: int) -> pl.DataFrame:
    cols = vocab["kind"].to_numpy() == kind
    idx = np.flatnonzero(cols)
    scores, top = probs[:, idx].topk(k, dim=-1)
    labels = vocab["label"].to_numpy()[idx]
    return pl.DataFrame(
        {
            "image": np.repeat(np.arange(probs.shape[0]), k),
            "label": labels[top.numpy().ravel()],
            "kind": kind,
            "score": scores.numpy().ravel(),
        }
    )


def top_labels_per_image(probs: torch.Tensor, ks: dict[str, int]) -> pl.DataFrame:
    frames = [top_k_for_kind(probs, kind, k) for kind, k in ks.items()]
    return pl.concat(frames).sort("image", "score", descending=[False, True])


siglip_probs = siglip_probabilities(image_emb, text_emb)
siglip_top_labels = top_labels_per_image(siglip_probs, siglip_top_k)
siglip_top_labels
```

## Check the SigLIP scores

Look at the score distribution. Then look at the labels on some images.

```python {.marimo}
def plot_score_distribution(top_labels: pl.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3))
    for kind in top_labels["kind"].unique():
        ax.hist(top_labels.filter(pl.col("kind") == kind)["score"], bins=40, alpha=0.6, label=kind)
    ax.set_xlabel("SigLIP score")
    ax.legend()
    return fig


plot_score_distribution(siglip_top_labels)
```

## Score the aesthetic quality of each image

Use MUSIQ. A higher score means a more aesthetic image.

```python {.marimo}
import tensorflow as tf
import tensorflow_hub as hub

musiq = hub.load("https://www.kaggle.com/models/google/musiq/tensorFlow2/ava/1")
musiq_fn = musiq.signatures["serving_default"]


def musiq_score(image_path: str) -> float:
    img_bytes = tf.constant(Path(image_path).read_bytes())
    return float(list(musiq_fn(img_bytes).values())[0].numpy())


aesthetic_scores = np.array([musiq_score(p) for p in amsterdam["path"]])
```

```python {.marimo}
def showcase_top_labels(top_labels: pl.DataFrame, dataset: pl.DataFrame,
                        n_images: int = 12, cols: int = 4):
    per_image = (
        top_labels.group_by("image", maintain_order=True)
        .agg(pl.col("label"), pl.col("score"), pl.col("kind"))
        .head(n_images)
    )

    rows = -(-per_image.height // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 5.5 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, row in zip(axes, per_image.iter_rows(named=True)):
        imgid = row["image"]
        path = dataset["path"][imgid]
        try:
            ax.imshow(Image.open(path).convert("RGB"))
        except Exception as e:
            ax.text(0.5, 0.5, f"failed to load\n{e}", ha="center", va="center", wrap=True)
        caption = "\n".join(
            f"[{k[0]}] {lbl} {s:.3f}"
            for lbl, s, k in zip(row["label"], row["score"], row["kind"])
        )
        ax.set_title(f"img {imgid}", fontsize=10, fontweight="bold")
        ax.set_xlabel(caption, fontsize=8, loc="left")
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[per_image.height:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


showcase_top_labels(siglip_top_labels, amsterdam, n_images=4, cols=4)
```

```python {.marimo}
def entities_per_image(top_labels: pl.DataFrame) -> dict[int, set[str]]:
    grouped = top_labels.group_by("image").agg(pl.col("label"))
    return {row["image"]: set(row["label"]) for row in grouped.iter_rows(named=True)}


def images_per_entity(entities: dict[int, set[str]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for image, labels in entities.items():
        for label in labels:
            index.setdefault(label, []).append(image)
    return index


def add_shared_entity_edges(graph: DiGraph, entities: dict[int, set[str]]) -> None:
    for label, images in images_per_entity(entities).items():
        for a in images:
            for b in images:
                if a != b:
                    add_shared_edge(graph, a, b, label)


def add_shared_edge(graph: DiGraph, a: int, b: int, label: str) -> None:
    if graph.has_edge(a, b):
        graph[a][b]["entities"].append(label)
        graph[a][b]["weight"] += 1
    else:
        graph.add_edge(a, b, entities=[label], weight=1)


def build_entity_graph(paths: list[str], entities: dict[int, set[str]],aesthetic_scores) -> DiGraph:
    graph = DiGraph()
    for i, path in enumerate(paths):
        graph.add_node(i, path=path, entities=entities.get(i, set()),aesthetic_score=aesthetic_scores[i])
    add_shared_entity_edges(graph, entities)
    return graph


image_entities = entities_per_image(siglip_top_labels)
graph = build_entity_graph(amsterdam["path"].to_list(), image_entities,aesthetic_scores)
print(graph.number_of_nodes(), graph.number_of_edges())
```

## Find paths - All simple paths with len k

```python {.marimo}
from itertools import filterfalse

paths = []
k=5
for i in graph.nodes:
    for j in graph.nodes:
        if i != j:
            asp = nx.all_simple_paths(graph, source=i, target=j,cutoff=k)
            paths.extend(asp)

paths_with_len_k=filterfalse(lambda p: len(p)==k,paths)
```

