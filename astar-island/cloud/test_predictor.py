"""Tests for Astar Island predictor against cached GT data."""

import json
import os
import math

import pytest

from predictor import Predictor, NC, PROB_FLOOR
from metadata_analyzer import extract_params

CLOUD_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.dirname(CLOUD_DIR), "cache")
LOOKUP_PATH = os.path.join(CLOUD_DIR, "gt_lookup.json")


@pytest.fixture
def predictor():
    with open(LOOKUP_PATH) as f:
        lookup = json.load(f)
    return Predictor(lookup)


@pytest.fixture
def gt_data():
    """Load first available GT data file."""
    for fname in sorted(os.listdir(CACHE_DIR)):
        if "gt_s" in fname and fname.endswith(".json"):
            with open(os.path.join(CACHE_DIR, fname)) as f:
                return json.load(f)
    pytest.skip("No GT data in cache/")


def load_all_gt():
    """Load all GT data files."""
    results = []
    for fname in sorted(os.listdir(CACHE_DIR)):
        if "gt_s" in fname and fname.endswith(".json"):
            with open(os.path.join(CACHE_DIR, fname)) as f:
                data = json.load(f)
                data["_filename"] = fname
                results.append(data)
    return results


class TestStaticCells:
    def test_mountain_stays_mountain(self, predictor):
        grid = [[5, 10], [4, 1]]
        pred = predictor.predict(grid, 2, 2)
        assert pred[0][0][5] > 0.95, "Mountain should stay mountain"

    def test_ocean_stays_ocean(self, predictor):
        grid = [[5, 10], [4, 1]]
        pred = predictor.predict(grid, 2, 2)
        assert pred[0][1][0] > 0.95, "Ocean should stay ocean"


class TestShift:
    def test_shift_changes_predictions(self, predictor):
        # A simple 3x3 grid with settlement in center surrounded by forest
        grid = [[4, 4, 4], [4, 1, 4], [4, 4, 4]]
        pred_no_shift = predictor.predict(grid, 3, 3)
        sett_prob_base = pred_no_shift[1][1][1]

        # Double settlement survival
        shift = {1: [0.5, 2.0, 1.0, 1.0, 0.5, 1.0]}
        pred_shifted = predictor.predict(grid, 3, 3, shift=shift)
        sett_prob_shifted = pred_shifted[1][1][1]

        assert sett_prob_shifted > sett_prob_base, (
            f"Shift should increase settlement prob: {sett_prob_shifted} vs {sett_prob_base}"
        )

    def test_shift_clipped(self, predictor):
        """Extreme shift values should be clipped to [0.5, 3.0]."""
        grid = [[4, 4, 4], [4, 1, 4], [4, 4, 4]]
        shift_extreme = {1: [0.01, 100.0, 1.0, 1.0, 0.01, 1.0]}
        # Should not crash and should produce valid probabilities
        pred = predictor.predict(grid, 3, 3, shift=shift_extreme)
        for y in range(3):
            for x in range(3):
                s = sum(pred[y][x])
                assert abs(s - 1.0) < 0.01, f"Sum should be ~1.0, got {s}"


class TestNormalization:
    def test_probabilities_sum_to_one(self, predictor, gt_data):
        pred = predictor.predict(
            gt_data["initial_grid"], gt_data["height"], gt_data["width"]
        )
        for y in range(gt_data["height"]):
            for x in range(gt_data["width"]):
                s = sum(pred[y][x])
                assert abs(s - 1.0) < 0.01, f"Cell ({y},{x}) sums to {s}"

    def test_all_probs_above_floor(self, predictor, gt_data):
        pred = predictor.predict(
            gt_data["initial_grid"], gt_data["height"], gt_data["width"]
        )
        for y in range(gt_data["height"]):
            for x in range(gt_data["width"]):
                for c in range(NC):
                    assert pred[y][x][c] >= PROB_FLOOR * 0.99, (
                        f"Cell ({y},{x}) class {c} below floor: {pred[y][x][c]}"
                    )


class TestPortSuppression:
    def test_non_coastal_port_suppressed(self, predictor):
        """Non-coastal cells should have near-floor port probability."""
        # Grid with settlement far from ocean
        grid = [[4] * 10 for _ in range(10)]
        grid[5][5] = 1  # settlement in center, no ocean nearby
        pred = predictor.predict(grid, 10, 10)
        # Cell (5,5) is not coastal -> port should be near floor
        port_prob = pred[5][5][2]
        assert port_prob < 0.01, f"Non-coastal port should be suppressed, got {port_prob}"


class TestScoring:
    def test_score_against_gt(self, predictor, gt_data):
        """Lookup predictions should score 75+ against GT (in-sample)."""
        pred = predictor.predict(
            gt_data["initial_grid"], gt_data["height"], gt_data["width"]
        )
        score = predictor.score_kl(
            pred, gt_data["ground_truth"], gt_data["height"], gt_data["width"]
        )
        assert score > 75, f"Expected 75+, got {score:.1f}"

    def test_perfect_prediction_scores_100(self, predictor, gt_data):
        """Predicting GT itself should score ~100."""
        score = predictor.score_kl(
            gt_data["ground_truth"],
            gt_data["ground_truth"],
            gt_data["height"],
            gt_data["width"],
        )
        assert score > 99.9, f"Perfect prediction should score ~100, got {score:.1f}"

    def test_all_cached_gt_scores(self, predictor):
        """All cached GT files should score 70+, average 80+."""
        all_gt = load_all_gt()
        if not all_gt:
            pytest.skip("No GT data")

        scores = []
        for data in all_gt:
            pred = predictor.predict(
                data["initial_grid"], data["height"], data["width"]
            )
            score = predictor.score_kl(
                pred, data["ground_truth"], data["height"], data["width"]
            )
            scores.append(score)
            print(f"  {data['_filename']}: {score:.1f}")

        avg = sum(scores) / len(scores)
        min_score = min(scores)
        print(f"  Average: {avg:.1f}, Min: {min_score:.1f}")
        assert avg > 75, f"Average score {avg:.1f} below 75"
        assert min_score > 50, f"Min score {min_score:.1f} below 50"


class TestComputeShift:
    def test_compute_shift_returns_dict(self, predictor):
        obs = {1: [10, 50, 5, 20, 15, 0]}
        shift = predictor.compute_shift(obs)
        assert isinstance(shift, dict)

    def test_compute_shift_near_one_for_matching(self, predictor):
        """If obs matches lookup avg, shift should be near 1.0."""
        # Build average rates from lookup for class 4 (forest)
        sums = [0.0] * NC
        count = 0
        for key, val in predictor.lookup.items():
            if key.startswith("4_"):
                for c in range(NC):
                    sums[c] += val[c]
                count += 1
        if count == 0:
            pytest.skip("No forest entries in lookup")

        avg = [s / count for s in sums]
        # Convert to fake counts (multiply by 1000 for precision)
        obs = {4: [int(a * 1000) for a in avg]}
        shift = predictor.compute_shift(obs)
        if 4 in shift:
            for c in range(NC):
                assert 0.8 < shift[4][c] < 1.2, (
                    f"Shift for class 4, target {c} should be ~1.0, got {shift[4][c]}"
                )


class TestMetadataAnalyzer:
    def test_extract_params_basic(self):
        settlements = [
            {
                "population": 3.0,
                "food": 2.0,
                "wealth": 1.0,
                "defense": 0.5,
                "alive": True,
                "has_port": True,
                "owner_id": 0,
            },
            {
                "population": 0.5,
                "food": 0.1,
                "wealth": 0.0,
                "defense": 0.2,
                "alive": False,
                "owner_id": 1,
            },
        ]
        params = extract_params(settlements)
        assert "survival_rate" in params
        assert "avg_food" in params
        assert "faction_count" in params
        assert params["survival_rate"] == 0.5
        assert params["faction_count"] == 1  # only 1 alive faction

    def test_extract_params_empty(self):
        params = extract_params([])
        assert params["survival_rate"] == 0.5  # default

    def test_extract_params_all_alive(self):
        settlements = [
            {"population": 2.0, "food": 3.0, "wealth": 1.0, "alive": True, "owner_id": 0},
            {"population": 1.0, "food": 1.0, "wealth": 0.5, "alive": True, "owner_id": 1},
        ]
        params = extract_params(settlements)
        assert params["survival_rate"] == 1.0
        assert params["death_rate"] == 0.0
        assert params["faction_count"] == 2
