import json
from pathlib import Path

from image_pipeline import ImageRecord
from models import DataModel

IMAGES_ROOT = Path("images")
VIST_IMAGES_ROOT = IMAGES_ROOT / "vist"
YFCC_IMAGES_ROOT = IMAGES_ROOT / "yfcc"
PERSONAL_PHOTOS_ROOT = Path("datasets/personal-photos")
PERSONAL_PHOTOS_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic")


def load_vist_dataset(json_path: Path) -> DataModel:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DataModel(**data)


def _images_in(album_dir: Path) -> list[ImageRecord]:
    if not album_dir.exists():
        return []
    return [ImageRecord(path=p, id=p.stem) for p in sorted(album_dir.glob("*.jpg"))]


def load_vist_album_images(
    album_id: str, json_path: Path | None = None
) -> list[ImageRecord]:
    album_dir = VIST_IMAGES_ROOT / album_id
    records = _images_in(album_dir)
    if records or json_path is None:
        return records

    from download_images import download_album

    download_album(load_vist_dataset(json_path), album_id)
    return _images_in(album_dir)


def load_personal_photos_images() -> list[ImageRecord]:
    if not PERSONAL_PHOTOS_ROOT.exists():
        return []
    paths = [
        p
        for p in sorted(PERSONAL_PHOTOS_ROOT.iterdir())
        if p.suffix.lower() in PERSONAL_PHOTOS_EXTENSIONS
    ]
    return [ImageRecord(path=p, id=p.stem) for p in paths]


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


CUFED_ROOT = Path("datasets/cufed")
LIVE_ROOT = Path("datasets/live/ChallengeDB_release")


def _cufed_rows(album_id: str, records: list, events: list[str]) -> list[dict]:
    return [
        {
            "event_id": album_id,
            "image_id": key.split("/")[1],
            "path": str(CUFED_ROOT / "images" / f"{key}.jpg"),
            "event_types": events,
            "context_importance": float(score),
            "flickr_url": url,
        }
        for url, key, score in records
    ]


def load_cufed(root: Path = CUFED_ROOT):
    import polars as pl

    events = json.loads((root / "event_type.json").read_text())
    importance = json.loads((root / "image_importance.json").read_text())
    album_ids = (root / "test_album_ids.txt").read_text().split()
    rows = [
        row
        for album_id in album_ids
        for row in _cufed_rows(album_id, importance[album_id], events[album_id])
    ]
    return pl.DataFrame(rows)


def _mat_vector(path: Path, key: str) -> list:
    from scipy.io import loadmat

    return loadmat(path)[key].ravel().tolist()


def _live_path(root: Path, name: str) -> str:
    folder = root / "Images"
    if name.startswith("t"):
        folder = folder / "trainingImages"
    return str(folder / name)


def load_live(root: Path = LIVE_ROOT):
    import polars as pl

    data = root / "Data"
    names = [
        str(n[0])
        for n in _mat_vector(data / "AllImages_release.mat", "AllImages_release")
    ]
    return pl.DataFrame(
        {
            "image_id": [Path(n).stem for n in names],
            "path": [_live_path(root, n) for n in names],
            "technical_quality": _mat_vector(
                data / "AllMOS_release.mat", "AllMOS_release"
            ),
            "quality_std": _mat_vector(
                data / "AllStdDev_release.mat", "AllStdDev_release"
            ),
            "is_training": [n.startswith("t") for n in names],
        }
    )
