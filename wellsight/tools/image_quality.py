"""Image quality check.

Runs before the detectors. A blurry, dark or glare-washed image makes every
later finding less trustworthy, so the agent uses this result to decide
whether to trust the scan or ask for a better image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import GREEN, RED, ToolResult, load_image, save_annotated

# Thresholds tuned on typical 1-2 MP phone and drone frames.
BLUR_MIN = 100.0        # variance of Laplacian below this = blurry
DARK_MAX = 60.0         # mean brightness (0-255) below this = too dark
BRIGHT_MIN = 215.0      # mean brightness above this = overexposed
GLARE_FRACTION = 0.08   # share of blown-out pixels above this = glare


def check_quality(image_path: str | Path, out_dir: str | Path | None = None) -> ToolResult:
    img = load_image(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    glare = float((gray >= 250).mean())

    issues: list[str] = []
    if sharpness < BLUR_MIN:
        issues.append("blurry")
    if brightness < DARK_MAX:
        issues.append("too_dark")
    if brightness > BRIGHT_MIN:
        issues.append("overexposed")
    if glare > GLARE_FRACTION:
        issues.append("glare")
    if contrast < 20:
        issues.append("low_contrast")

    # Score: start at 1 and subtract per problem, weighted by how bad it is.
    score = 1.0
    if "blurry" in issues:
        score -= 0.5 * (1 - sharpness / BLUR_MIN)
    if "too_dark" in issues:
        score -= 0.2 + 0.6 * (1 - brightness / DARK_MAX)
    if "overexposed" in issues:
        score -= 0.3
    if "glare" in issues:
        score -= min(0.3, glare * 2)
    if "low_contrast" in issues:
        score -= 0.15
    score = round(max(0.0, score), 3)

    usable = score >= 0.6
    canvas = img.copy()
    color = GREEN if usable else RED
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 6)
    text = f"quality {score:.2f} " + (", ".join(issues) if issues else "ok")
    cv2.putText(canvas, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    return ToolResult(
        tool="image_quality",
        image=str(image_path),
        width=w,
        height=h,
        summary={
            "quality_score": score,
            "usable": usable,
            "issues": issues,
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "glare_fraction": round(glare, 4),
            "recommendation": "proceed" if usable else "request_new_image",
        },
        annotated_path=save_annotated(canvas, image_path, "quality", out_dir),
    )
