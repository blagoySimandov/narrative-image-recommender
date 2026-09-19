import json
from pathlib import Path

from image_pipeline import ImageRecord
from models import DataModel

IMAGES_ROOT = Path("images")
VIST_IMAGES_ROOT = IMAGES_ROOT / "vist"
YFCC_IMAGES_ROOT = IMAGES_ROOT / "yfcc"


def load_vist_dataset(json_path: Path) -> DataModel:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DataModel(**data)


def _images_in(album_dir: Path) -> list[ImageRecord]:
    if not album_dir.exists():
        return []
    return [ImageRecord(path=p, id=p.stem) for p in sorted(album_dir.glob("*.jpg"))]


def load_vist_album_images(
    album_id: str, model: DataModel | None = None
) -> list[ImageRecord]:
    album_dir = VIST_IMAGES_ROOT / album_id
    records = _images_in(album_dir)
    if records or model is None:
        return records

    from download_images import download_album

    download_album(model, album_id)
    return _images_in(album_dir)


def load_yfcc_album_images(
    city: str, album_id: str, city_dir: Path | None = None
) -> list[ImageRecord]:
    album_dir = YFCC_IMAGES_ROOT / city / album_id
    records = _images_in(album_dir)
    if records or city_dir is None:
        return records

    from download_yfcc_images import download_album

    download_album(city, city_dir, album_id)
    return _images_in(album_dir)
