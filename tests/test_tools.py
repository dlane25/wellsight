"""Behavior checks for the vision tools on synthetic scenes.

Run:  python -m pytest -q
"""
import pytest

from tests.make_samples import make_all
from wellsight.registry import run_tool


@pytest.fixture(scope="session")
def samples():
    return {k: str(v) for k, v in make_all().items()}


def summary(tool, path):
    return run_tool(tool, path)["summary"]


def test_quality_flags_bad_images(samples):
    assert summary("check_image_quality", samples["pad_spill"])["usable"]
    blurry = summary("check_image_quality", samples["pad_spill_blurry"])
    assert not blurry["usable"] and "blurry" in blurry["issues"]
    dark = summary("check_image_quality", samples["pad_dark"])
    assert not dark["usable"] and "too_dark" in dark["issues"]


def test_spill_found_on_stained_pad(samples):
    s = summary("detect_spills", samples["pad_spill"])
    assert s["spill_detected"] and s["max_confidence"] >= 0.7
    assert s["stained_fraction"] > 0.04


def test_no_spill_on_clean_pad(samples):
    assert not summary("detect_spills", samples["pad_clean"])["spill_detected"]


def test_corrosion_grades_in_order(samples):
    grades = [summary("score_corrosion", samples[k])["corrosion_grade"]
              for k in ("tank_clean", "tank_rust_mild", "tank_rust_severe")]
    assert grades == ["none", "mild", "severe"]


def test_results_are_json_ready(samples):
    import json
    json.dumps(run_tool("detect_spills", samples["pad_spill"]))
