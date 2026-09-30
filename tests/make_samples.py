"""Generate synthetic test scenes so the tools can be checked before real photos arrive.

These are stand-ins, not evaluation data. Replace with the Roboflow sets and
your own pad photos for the real evaluation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def gravel(h: int = 720, w: int = 960, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (150, 165, 175), np.float32)          # tan-gray (BGR)
    base += rng.normal(0, 18, (h, w, 1))                            # grain
    coarse = cv2.resize(rng.normal(0, 14, (h // 12, w // 12)).astype(np.float32), (w, h))
    base += coarse[..., None]
    for _ in range(1400):                                           # individual stones
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(2, 7))
        shade = float(rng.normal(0, 30))
        cv2.circle(base, (x, y), r, (150 + shade, 160 + shade, 170 + shade), -1)
    return np.clip(base, 0, 255).astype(np.uint8)


def add_stain(img: np.ndarray, center: tuple[int, int], axes: tuple[int, int], seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, center, axes, 20, 0, 360, 1.0, -1)
    for _ in range(8):                                              # irregular lobes
        dx, dy = rng.integers(-axes[0], axes[0]), rng.integers(-axes[1], axes[1])
        cv2.circle(mask, (center[0] + int(dx), center[1] + int(dy)), int(rng.integers(20, 50)), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
    stained = img.astype(np.float32) * 0.35 + 18                    # dark, flattened texture
    stained = cv2.GaussianBlur(stained, (7, 7), 0)
    out = img.astype(np.float32) * (1 - mask) + stained * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def rusty_plate(h: int = 720, w: int = 960, amount: float = 0.2, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    plate = np.full((h, w, 3), (145, 140, 135), np.float32)          # painted steel gray
    plate += rng.normal(0, 6, (h, w, 1))
    cv2.rectangle(plate, (0, h // 2 - 4), (w, h // 2 + 4), (110, 108, 105), -1)   # weld seam
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 18)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    rust = (noise > 1 - amount).astype(np.float32)
    rust = cv2.GaussianBlur(rust, (9, 9), 0)[..., None]
    rust_color = np.full((h, w, 3), (30, 75, 150), np.float32) + rng.normal(0, 20, (h, w, 1))
    out = plate * (1 - rust) + rust_color * rust
    return np.clip(out, 0, 255).astype(np.uint8)


def make_all() -> dict[str, Path]:
    SAMPLES.mkdir(exist_ok=True)
    g = gravel()
    scenes = {
        "pad_clean": g,
        "pad_spill": add_stain(g, (520, 380), (150, 90)),
        "pad_spill_blurry": cv2.GaussianBlur(add_stain(g, (520, 380), (150, 90)), (0, 0), 6),
        "pad_dark": (add_stain(g, (520, 380), (150, 90)) * 0.2).astype(np.uint8),
        "tank_clean": rusty_plate(amount=0.0),
        "tank_rust_mild": rusty_plate(amount=0.12),
        "tank_rust_severe": rusty_plate(amount=0.35),
    }
    paths = {}
    for name, img in scenes.items():
        p = SAMPLES / f"{name}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths[name] = p
    return paths


if __name__ == "__main__":
    for name, p in make_all().items():
        print(name, p)
