from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"


def download(url: str, destination: Path, *, skip_existing: bool) -> None:
    if skip_existing and destination.exists():
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TinyStories V2 GPT-4 text files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/tinystories/raw"))
    parser.add_argument("--split", choices=["train", "valid", "both"], default="both")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits = ["train", "valid"] if args.split == "both" else [args.split]
    for split in splits:
        filename = FILES[split]
        download(f"{BASE_URL}/{filename}", args.out_dir / filename, skip_existing=not args.overwrite)


if __name__ == "__main__":
    main()
