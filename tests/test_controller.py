import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import FuzzyStreetlightController, scenario_rows, threshold_controller


def test_brightness_stays_in_valid_range_for_grid():
    controller = FuzzyStreetlightController()
    for lux in range(0, 1001, 50):
        for traffic in range(0, 101, 10):
            value, strengths = controller.evaluate(lux, traffic)
            assert 0 <= value <= 100
            assert all(0 <= strength <= 1 for strength in strengths.values())


def test_heavy_traffic_in_darkness_requires_more_light():
    controller = FuzzyStreetlightController()
    low, _ = controller.evaluate(50, 10)
    high, _ = controller.evaluate(50, 90)
    assert high > low


def test_bright_daylight_is_low_brightness():
    controller = FuzzyStreetlightController()
    low_traffic, _ = controller.evaluate(950, 10)
    high_traffic, _ = controller.evaluate(950, 90)
    assert low_traffic < 20
    assert high_traffic > low_traffic


def test_scenario_table_contains_all_five_cases():
    rows = list(scenario_rows(FuzzyStreetlightController()))
    assert len(rows) == 5
    assert all(0 <= row["Fuzzy brightness (%)"] <= 100 for row in rows)


def test_threshold_baseline_has_expected_fixed_behavior():
    assert threshold_controller(950, 10) == 0
    assert threshold_controller(950, 90) == 20
    assert threshold_controller(50, 90) == 100
