import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import requests

DATASETS_ROOT = Path("datasets")
IMAGES_ROOT = Path("images") / "yfcc"
FAILED_LOG = Path("failed_downloads.log")
TIMEOUT = 10
MAX_WORKERS = 16


def find_album_csvs(city_dir: Path, max_albums: Optional[int]) -> list[Path]:
    csv_paths = sorted(city_dir.glob("*.csv"), key=lambda p: int(p.stem))
    if max_albums is None:
        return csv_paths
    return csv_paths[:max_albums]


def read_album_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return list(reader)


def download_one(city: str, album_id: str, row: dict) -> Tuple[str, bool, str]:
    photo_id = row["id"]
    url = row["url_s"]
    if not url:
        return (photo_id, False, "no url")

    dest_dir = IMAGES_ROOT / city / album_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{photo_id}.jpg"

    if dest_path.exists():
        return (photo_id, True, "already exists")

    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return (photo_id, True, "ok")
    except requests.RequestException as exc:
        return (photo_id, False, str(exc))


def download_all(city: str, albums: dict) -> None:
    tasks = [
        (album_id, row)
        for album_id, rows in albums.items()
        for row in rows
    ]
    total = len(tasks)
    done = 0
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_one, city, album_id, row): row["id"]
            for album_id, row in tasks
        }
        for future in as_completed(futures):
            photo_id, ok, message = future.result()
            done += 1
            if not ok:
                failures.append(f"{photo_id}\t{message}")
            if done % 200 == 0:
                print(f"{done}/{total} processed, {len(failures)} failed")

    if failures:
        FAILED_LOG.write_text("\n".join(failures), encoding="utf-8")
        print(f"Done. {len(failures)} failures logged to {FAILED_LOG}")
    else:
        print("Done. No failures.")


def download_album(city: str, city_dir: Path, album_id: str) -> None:
    csv_path = city_dir / f"{album_id}.csv"
    rows = read_album_rows(csv_path)
    download_all(city, {album_id: rows})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("city_dir", type=Path)
    parser.add_argument("--max-albums", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    city = args.city_dir.name.split("-")[-1].capitalize()

    csv_paths = find_album_csvs(args.city_dir, args.max_albums)
    albums = {csv_path.stem: read_album_rows(csv_path) for csv_path in csv_paths}

    total_images = sum(len(rows) for rows in albums.values())
    print(f"{len(albums)} albums, {total_images} images. Starting download.")
    download_all(city, albums)


if __name__ == "__main__":
    main()
