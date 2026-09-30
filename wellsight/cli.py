"""Command line entry point.

    python -m wellsight.cli path/to/image.jpg --out results/
    python -m wellsight.cli path/to/folder --tool detect_spills --out results/

Prints one JSON result per image and tool, and writes annotated images to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .registry import TOOLS, run_tool

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _images(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [target]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run WellSight OpenCV 5 tools on site images.")
    ap.add_argument("target", type=Path, help="An image file or a folder of images")
    ap.add_argument("--tool", choices=list(TOOLS), action="append",
                    help="Tool to run (repeatable). Default: all tools.")
    ap.add_argument("--out", default=None, help="Folder for annotated images and results.json")
    args = ap.parse_args(argv)

    tools = args.tool or list(TOOLS)
    results = []
    for img in _images(args.target):
        for name in tools:
            results.append(run_tool(name, str(img), args.out))

    output = {"opencv_version": cv2.__version__, "results": results}
    text = json.dumps(output, indent=2)
    print(text)
    if args.out:
        Path(args.out, "results.json").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
