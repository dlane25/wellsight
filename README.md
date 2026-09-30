# WellSight

Agentic visual inspection for oil and gas wellsites and pipelines, built for the
OpenCV AI Competition 2026. This repo currently holds the first three OpenCV 5
vision tools. The agent loop, MCP server and AWS deployment come next.

## Tools

| Tool | What it finds | How |
| --- | --- | --- |
| `check_image_quality` | Blur, darkness, overexposure, glare | Laplacian variance, brightness and histogram stats |
| `detect_spills` | Oil and liquid stains on gravel or soil | Lab lightness vs. a closing-based background estimate, saturation gate, texture-flattening and edge-shape scoring |
| `score_corrosion` | Rust on tanks, valves, flanges, pipe | HSV rust band, Lab redness, edge density; graded none / mild / moderate / severe |

Every tool returns JSON the agent reasons over (findings with confidence, bounding
box, severity, details, plus a summary) and an annotated image for humans. Tool
descriptions written for the agent live in `wellsight/registry.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Pinned to OpenCV 5.0.0.

## Run

```bash
python -m tests.make_samples                    # generate synthetic test scenes into samples/
python -m wellsight.cli samples --out out       # all tools on every image
python -m wellsight.cli photo.jpg --tool detect_spills --out out
python -m pytest -q                             # behavior checks
```

Results print as JSON and are saved to `out/results.json` with annotated images
beside them.

## Current results on synthetic scenes

| Scene | Quality | Spill | Corrosion |
| --- | --- | --- | --- |
| Clean pad | usable | none | none |
| Pad with stain | usable | detected, 0.83 | none |
| Same stain, blurred | not usable (blurry) | detected, 0.70 | none |
| Same stain, very dark | not usable (too dark) | missed | none |
| Clean tank | usable | none | none |
| Tank, light rust | usable | none | mild |
| Tank, heavy rust | usable | none | severe |

The dark case is a deliberate failure: the quality tool flags it so the agent
asks for a new image instead of trusting the miss.

## Known limitations

- Hard shadows can look like stains. The texture and edge checks lower their
  confidence; the agent should ask for a second angle when confidence is 0.35-0.6.
- Rust detection is color-based, so orange paint or clay soil can trigger it.
  The planned DNN equipment detector will restrict it to metal surfaces.
- Thresholds were tuned on synthetic scenes only. Tune them on the Roboflow
  Rust Detection and Oil Spillage sets and real pad photos before evaluation.

## Next steps

1. Download the Roboflow sets and write `eval/` scripts for precision and recall
2. Change detection (feature matching + homography) and zoom-and-reanalyze tools
3. MCP server exposing the registry, then the Bedrock agent loop
