"""
Resize images in `faces_dataset_small/class0` from 1024x1024 to 256x256.

Usage:
    python compress_images.py

Optional arguments:
    python compress_images.py --source "E:\\data_diffuse\\datasets\\faces_dataset_small\\class0" \
                              --destination "E:\\data_diffuse\\datasets\\faces_dataset_small\\class0_256" \
                              --size 256
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

from PIL import Image


DEFAULT_SOURCE = pathlib.Path(
    r"E:\data_diffuse\datasets\faces_dataset_small\huge"
)
DEFAULT_DESTINATION = DEFAULT_SOURCE.with_name(f"{DEFAULT_SOURCE.name}_256")


def iter_image_files(directory: pathlib.Path) -> Iterable[pathlib.Path]:
    supported_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for path in directory.rglob("*"):
        if path.suffix.lower() in supported_suffixes:
            yield path


def resize_image(
    source_path: pathlib.Path, destination_path: pathlib.Path, size: int
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        resized = image.resize((size, size), Image.LANCZOS)
        resized.save(destination_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize dataset images.")
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=DEFAULT_SOURCE,
        help="Directory with original images.",
    )
    parser.add_argument(
        "--destination",
        type=pathlib.Path,
        default=DEFAULT_DESTINATION,
        help="Directory for resized images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Target square size in pixels.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {args.source}"
        )
    if not args.source.is_dir():
        raise NotADirectoryError(
            f"Source path is not a directory: {args.source}"
        )

    image_paths = list(iter_image_files(args.source))
    if not image_paths:
        print(f"No supported image files found in {args.source}")
        return

    print(
        f"Resizing {len(image_paths)} images from {args.source} "
        f"to {args.destination} at {args.size}x{args.size}px"
    )

    for path in image_paths:
        relative = path.relative_to(args.source)
        destination_path = args.destination / relative
        resize_image(path, destination_path, args.size)

    print("Done.")


if __name__ == "__main__":
    main()

