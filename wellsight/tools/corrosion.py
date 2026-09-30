"""Corrosion and rust scoring.

Rust on tanks, valves and pipe falls in a narrow orange-to-brown hue band with
clear saturation, and it roughens the surface texture. This tool segments
that color band, merges nearby patches into regions, and grades each region
and the whole image as none / mild / moderate / severe. Those grades match
the labels in the public Rust Detection dataset so results can be scored
against it directly.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import ORANGE, RED, YELLOW, Finding, ToolResult, clamp01, draw_finding, load_image, save_annotated

# OpenCV hue runs 0-179. Rust sits roughly between red and orange-yellow.
BRIGHT_RUST = ((4, 80, 60), (22, 255, 235))
DARK_RUST = ((0, 60, 30), (18, 255, 120))
MIN_AREA_FRACTION = 0.001

# Share of the image covered by rust -> overall grade
GRADES = [(0.003, "none"), (0.03, "mild"), (0.12, "moderate"), (1.01, "severe")]
SEVERITY_FOR_GRADE = {"none": "info", "mild": "low", "moderate": "medium", "severe": "high"}


def _grade(fraction: float) -> str:
    for limit, name in GRADES:
        if fraction < limit:
            return name
    return "severe"


def score_corrosion(image_path: str | Path, out_dir: str | Path | None = None) -> ToolResult:
    img = load_image(image_path)
    h, w = img.shape[:2]
    total = h * w

    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    redness = lab[:, :, 1].astype(np.float32) - 128  # Lab a*: positive = toward red

    mask = cv2.inRange(hsv, *BRIGHT_RUST) | cv2.inRange(hsv, *DARK_RUST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Join speckled rust into regions so one flaking area is one finding
    regions = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    findings: list[Finding] = []
    canvas = img.copy()
    overlay = canvas.copy()
    overlay[mask > 0] = (0, 90, 255)
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    for c in contours:
        region_area = int(cv2.contourArea(c))
        if region_area < MIN_AREA_FRACTION * total:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        region = np.zeros((h, w), np.uint8)
        cv2.drawContours(region, [c], -1, 255, -1)
        region_area = int(np.count_nonzero(region))   # exact pixel count; contourArea undercounts small regions

        rust_px = int(np.count_nonzero(mask[region > 0]))
        density = rust_px / max(region_area, 1)                  # how solid the rust is inside the region
        mean_red = float(redness[mask & region > 0].mean()) if rust_px else 0.0
        roughness = float(np.count_nonzero(edges[region > 0])) / max(region_area, 1)

        frac = rust_px / total
        grade = _grade(frac * 4)   # a single region is graded more strictly than the whole image
        confidence = clamp01(
            0.40 * clamp01(density / 0.6)
            + 0.30 * clamp01(mean_red / 25)
            + 0.20 * clamp01(roughness / 0.08)
            + 0.10 * clamp01(frac / 0.02)
        )
        f = Finding(
            kind="corrosion",
            confidence=round(confidence, 3),
            bbox=[int(x), int(y), int(bw), int(bh)],
            area_px=rust_px,
            severity=SEVERITY_FOR_GRADE[grade] if grade != "none" else "low",
            details={
                "grade": grade if grade != "none" else "mild",
                "rust_fraction_of_image": round(frac, 4),
                "density": round(density, 3),
                "redness": round(mean_red, 1),
                "surface_roughness": round(roughness, 3),
            },
        )
        findings.append(f)
        draw_finding(canvas, f, {"high": RED, "medium": ORANGE, "low": YELLOW}.get(f.severity, YELLOW))

    findings.sort(key=lambda f: f.confidence, reverse=True)
    coverage = float(np.count_nonzero(mask)) / total
    overall = _grade(coverage)

    return ToolResult(
        tool="corrosion_scoring",
        image=str(image_path),
        width=w,
        height=h,
        findings=findings,
        summary={
            "corrosion_grade": overall,
            "rust_coverage": round(coverage, 4),
            "region_count": len(findings),
            "max_confidence": max((f.confidence for f in findings), default=0.0),
        },
        annotated_path=save_annotated(canvas, image_path, "corrosion", out_dir),
    )
