"""Spill and ground-stain detection.

Oil and produced-water stains on a pad show up as patches that are darker and
smoother than the gravel or soil around them, and usually low in color
saturation. This tool compares each pixel to its local background rather than
to a fixed threshold, so it keeps working as lighting changes across a site.

Known limitation: hard shadows are also dark and desaturated. Shadows tend to
have sharp straight edges and keep the ground texture; the texture check below
lowers their confidence but does not rule them out. The agent can ask for a
second image from another angle when confidence is middling.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import ORANGE, RED, YELLOW, Finding, ToolResult, clamp01, draw_finding, load_image, save_annotated

DARKNESS_DELTA = 28     # how much darker than local background (L channel, 0-255)
MAX_SATURATION = 110    # stains are mostly low-saturation
MIN_AREA_FRACTION = 0.0015


def _local_background(channel: np.ndarray) -> np.ndarray:
    """Estimate what the ground would look like with no stain.

    A grayscale closing fills in dark patches smaller than its kernel with the
    brightness around them, so even a large stain gets a clean background
    estimate. Done at 1/8 resolution for speed.
    """
    h, w = channel.shape
    small = cv2.resize(channel, (max(16, w // 8), max(16, h // 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    k = max(5, (min(small.shape) // 2) | 1)
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Closing picks up the brightest stones; pull back toward the typical level.
    median = float(np.median(small))
    closed = closed.astype(np.float32)
    closed = np.where(closed > median, median + 0.6 * (closed - median), closed)
    closed = cv2.GaussianBlur(closed, (0, 0), 3).astype(np.uint8)
    return cv2.resize(closed, (w, h), interpolation=cv2.INTER_LINEAR)


def _local_std(gray: np.ndarray, k: int = 9) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sq = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def detect_spills(image_path: str | Path, out_dir: str | Path | None = None,
                  darkness_delta: int = DARKNESS_DELTA) -> ToolResult:
    img = load_image(image_path)
    h, w = img.shape[:2]
    total = h * w

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    L = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)
    sat = hsv[:, :, 1]

    background = _local_background(L)
    darkness = cv2.subtract(background, L)          # positive where darker than surroundings

    mask = ((darkness > darkness_delta) & (sat < MAX_SATURATION)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    texture = _local_std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    findings: list[Finding] = []
    canvas = img.copy()

    for c in contours:
        area = int(cv2.contourArea(c))
        if area < MIN_AREA_FRACTION * total:
            continue
        x, y, bw, bh = cv2.boundingRect(c)

        region = np.zeros((h, w), np.uint8)
        cv2.drawContours(region, [c], -1, 255, -1)
        ring = cv2.subtract(cv2.dilate(region, ring_kernel), region)

        mean_dark = float(darkness[region > 0].mean())
        inside_tex = float(texture[region > 0].mean())
        outside_tex = float(texture[ring > 0].mean()) if ring.any() else inside_tex
        smoothness = clamp01(1 - inside_tex / (outside_tex + 1e-6))   # stains flatten the texture

        perim = cv2.arcLength(c, True)
        irregularity = clamp01((perim * perim) / (4 * np.pi * area) / 4)  # organic edges > straight edges

        area_frac = area / total
        confidence = clamp01(
            0.45 * clamp01(mean_dark / 70)
            + 0.30 * smoothness
            + 0.15 * irregularity
            + 0.10 * clamp01(area_frac / 0.03)
        )
        severity = "high" if area_frac > 0.05 else "medium" if area_frac > 0.01 else "low"

        f = Finding(
            kind="spill",
            confidence=round(confidence, 3),
            bbox=[int(x), int(y), int(bw), int(bh)],
            area_px=area,
            severity=severity,
            details={
                "area_fraction": round(area_frac, 4),
                "mean_darkness": round(mean_dark, 1),
                "smoothness": round(smoothness, 3),
                "edge_irregularity": round(irregularity, 3),
            },
        )
        findings.append(f)
        cv2.drawContours(canvas, [c], -1, RED, 2)
        draw_finding(canvas, f, {"high": RED, "medium": ORANGE, "low": YELLOW}[severity])

    findings.sort(key=lambda f: f.confidence, reverse=True)
    covered = sum(f.area_px for f in findings) / total

    return ToolResult(
        tool="spill_detection",
        image=str(image_path),
        width=w,
        height=h,
        findings=findings,
        summary={
            "spill_detected": any(f.confidence >= 0.5 for f in findings),
            "max_confidence": max((f.confidence for f in findings), default=0.0),
            "stained_fraction": round(covered, 4),
            "uncertain_regions": sum(1 for f in findings if 0.35 <= f.confidence < 0.6),
        },
        annotated_path=save_annotated(canvas, image_path, "spill", out_dir),
    )
