"""Shared types and helpers for WellSight vision tools.

Every tool returns a ToolResult: structured findings the agent can reason
over, plus an annotated image a human can check. The agent never sees raw
pixels, only these results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# BGR colors for annotations
RED = (40, 40, 220)
ORANGE = (0, 140, 255)
YELLOW = (0, 215, 255)
GREEN = (60, 180, 60)


@dataclass
class Finding:
    kind: str                 # e.g. "spill", "corrosion"
    confidence: float         # 0.0 - 1.0
    bbox: list[int]           # [x, y, w, h] in pixels
    area_px: int
    severity: str = "info"    # info | low | medium | high
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool: str
    image: str
    width: int
    height: int
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    annotated_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["finding_count"] = len(self.findings)
        return d


def load_image(path: str | Path, max_side: int = 1600) -> np.ndarray:
    """Read an image and downscale very large ones so every tool runs fast."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def draw_finding(canvas: np.ndarray, f: Finding, color: tuple[int, int, int]) -> None:
    x, y, w, h = f.bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    label = f"{f.kind} {f.severity} {f.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y - 6, th + 4)
    cv2.rectangle(canvas, (x, ty - th - 4), (x + tw + 6, ty + 2), color, -1)
    cv2.putText(canvas, label, (x + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def save_annotated(canvas: np.ndarray, image_path: str | Path, tool: str,
                   out_dir: str | Path | None) -> str | None:
    if out_dir is None:
        return None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(image_path).stem}_{tool}.jpg"
    cv2.imwrite(str(path), canvas)
    return str(path)


def clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))
