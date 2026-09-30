"""Tool registry.

One place that lists every vision tool with a plain description and input
schema. The MCP server (next step) will expose these exact entries to the
agent, so the descriptions are written for the model that calls them.
"""
from __future__ import annotations

from typing import Any, Callable

from .tools.corrosion import score_corrosion
from .tools.image_quality import check_quality
from .tools.spill import detect_spills

_IMAGE_INPUT = {
    "type": "object",
    "properties": {"image_path": {"type": "string", "description": "Path or S3 key of the site image"}},
    "required": ["image_path"],
}

TOOLS: dict[str, dict[str, Any]] = {
    "check_image_quality": {
        "fn": check_quality,
        "description": (
            "Check whether a site image is sharp, well lit and free of glare. Run this first. "
            "If 'usable' is false, do not trust detector results; request a new image instead."
        ),
        "input_schema": _IMAGE_INPUT,
    },
    "detect_spills": {
        "fn": detect_spills,
        "description": (
            "Find oil or liquid stains on the ground: patches darker and smoother than the "
            "surrounding gravel or soil. Returns each region with confidence and severity. "
            "Confidence between 0.35 and 0.6 means uncertain; hard shadows can look similar."
        ),
        "input_schema": _IMAGE_INPUT,
    },
    "score_corrosion": {
        "fn": score_corrosion,
        "description": (
            "Find rust on tanks, valves, flanges and pipe. Returns rust regions and an overall "
            "grade: none, mild, moderate or severe."
        ),
        "input_schema": _IMAGE_INPUT,
    },
}


def run_tool(name: str, image_path: str, out_dir: str | None = None) -> dict[str, Any]:
    if name not in TOOLS:
        raise KeyError(f"Unknown tool: {name}. Available: {', '.join(TOOLS)}")
    fn: Callable = TOOLS[name]["fn"]
    return fn(image_path, out_dir=out_dir).to_dict()
