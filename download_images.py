import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import requests

from models import DataModel

IMAGES_ROOT = Path("images")
FAILED_LOG = Path("failed_downloads.log")
TIMEOUT = 10
MAX_WORKERS = 16


def load_model(json_path: Path) -> DataModel:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DataModel(**data)


def pick_url(image) -> Optional[str]:
    return image.url_o or image.url_m


def dedupe_images(model: DataModel) -> dict:
    unique = {}
    for image in model.images:
        unique[image.id] = image
    return unique


def select_albums(model: DataModel, max_albums: Optional[int]) -> set:
    if max_albums is None:
        return {album.id for album in model.albums}
    return {album.id for album in model.albums[:max_albums]}


def filter_by_albums(images: dict, album_ids: set) -> dict:
    return {
        image_id: image
        for image_id, image in images.items()
        if image.album_id in album_ids
    }


def download_one(image) -> Tuple[str, bool, str]:
    url = pick_url(image)
    if url is None:
        return (image.id, False, "no url")

    dest_dir = IMAGES_ROOT / image.album_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{image.id}.jpg"

    if dest_path.exists():
        return (image.id, True, "already exists")

    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return (image.id, True, "ok")
    except requests.RequestException as exc:
        return (image.id, False, str(exc))


def download_all(images: dict) -> None:
    failures = []
    total = len(images)
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, img): img for img in images.values()}
        for future in as_completed(futures):
            image_id, ok, message = future.result()
            done += 1
            if not ok:
                failures.append(f"{image_id}\t{message}")
            if done % 200 == 0:
                print(f"{done}/{total} processed, {len(failures)} failed")

    if failures:
        FAILED_LOG.write_text("\n".join(failures), encoding="utf-8")
        print(f"Done. {len(failures)} failures logged to {FAILED_LOG}")
    else:
        print("Done. No failures.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--max-albums", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.json_path)
    images = dedupe_images(model)

    album_ids = select_albums(model, args.max_albums)
    images = filter_by_albums(images, album_ids)

    print(f"{len(album_ids)} albums, {len(images)} unique images. Starting download.")
    download_all(images)


if __name__ == "__main__":
    main()
