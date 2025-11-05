from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


DATASETS = {
    "monet": "https://people.eecs.berkeley.edu/~tinghuiz/projects/cyclegan/datasets/monet2photo.zip",
    "cezanne": "https://people.eecs.berkeley.edu/~tinghuiz/projects/cyclegan/datasets/cezanne2photo.zip",
    "vangogh": "https://people.eecs.berkeley.edu/~tinghuiz/projects/cyclegan/datasets/vangogh2photo.zip",
}

PRETRAINED_URLS = {
    "monet": "https://huggingface.co/camenduru/autopainter-monet/resolve/main/autopainter_monet_G.pt",
}


def download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        print(f"[skip] {target_path} already exists")
        return

    print(f"Downloading {url} -> {target_path}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with target_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(archive: Path, target_dir: Path) -> None:
    print(f"Extracting {archive} -> {target_dir}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(target_dir)


def prepare_dataset(raw_dir: Path, processed_root: Path, style: str, val_split: float) -> None:
    source_root = raw_dir / f"{style}2photo"
    if not source_root.exists():
        alt_root = raw_dir / style
        if alt_root.exists():
            source_root = alt_root
        else:
            raise FileNotFoundError(f"Raw dataset for {style} not found under {source_root}")

    train_a = list((source_root / "trainA").glob("*"))
    test_a = list((source_root / "testA").glob("*"))
    if not train_a:
        raise RuntimeError(f"No training images found for {style}")

    processed_style_dir = processed_root / style
    train_dir = processed_style_dir / "train"
    val_dir = processed_style_dir / "val"
    test_dir = processed_style_dir / "test"
    for directory in (train_dir, val_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    import random

    random.seed(42)
    random.shuffle(train_a)
    val_count = max(1, int(len(train_a) * val_split))
    val_samples = train_a[:val_count]
    train_samples = train_a[val_count:]

    def _copy(samples: Iterable[Path], destination: Path) -> None:
        for path in tqdm(list(samples), desc=f"{style}:{destination.name}"):
            target_path = destination / path.name
            if target_path.exists():
                continue
            shutil.copy2(path, target_path)

    _copy(train_samples, train_dir)
    _copy(val_samples, val_dir)
    _copy(test_a, test_dir)


def download_pretrained(style: str, target_dir: Path) -> None:
    url = PRETRAINED_URLS.get(style)
    if not url:
        print(f"No pretrained weights registered for style '{style}'")
        return

    target = target_dir / f"autopainter_{style}_generator.pt"
    download_file(url, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets and pretrained weights for AutoPainter")
    parser.add_argument("--styles", nargs="*", default=["monet"], help="Styles to download (subset of monet, cezanne, vangogh)")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Data root directory")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--skip-pretrained", action="store_true", help="Skip downloading pretrained weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.data_root / "raw"
    processed_root = args.data_root / "processed" / "monet_sketches"

    for style in args.styles:
        style = style.lower()
        if style not in DATASETS:
            raise ValueError(f"Unknown style '{style}'")

        url = DATASETS[style]
        archive_path = raw_root / f"{style}.zip"
        download_file(url, archive_path)
        extract_zip(archive_path, raw_root)
        prepare_dataset(raw_root, processed_root, style, args.val_split)

        if not args.skip_pretrained:
            download_pretrained(style, Path("checkpoints") / f"autopainter_{style}")


if __name__ == "__main__":
    main()
