"""Interactive fuzzy streetlight controller.

The controller is deliberately kept in this single file so it is easy to
demonstrate in a college mini-project.  The core engine uses NumPy arrays and
implements Mamdani inference explicitly: fuzzification, min rule firing,
max aggregation, and centroid defuzzification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import skfuzzy as fuzz


LUX_UNIVERSE = np.arange(0, 1001, 1, dtype=float)
TRAFFIC_UNIVERSE = np.arange(0, 101, 1, dtype=float)
BRIGHTNESS_UNIVERSE = np.arange(0, 101, 1, dtype=float)


def trimf(universe: np.ndarray, points: Tuple[float, float, float]) -> np.ndarray:
    """Return a triangular membership function, including shoulder triangles."""
    a, b, c = points
    result = np.zeros_like(universe, dtype=float)
    if b > a:
        rising = (universe >= a) & (universe <= b)
        result[rising] = (universe[rising] - a) / (b - a)
    else:
        result[universe == b] = 1.0
    if c > b:
        falling = (universe >= b) & (universe <= c)
        result[falling] = np.maximum(result[falling], (c - universe[falling]) / (c - b))
    else:
        result[universe == b] = 1.0
    result[universe == b] = 1.0
    return np.clip(result, 0.0, 1.0)


def trapmf(universe: np.ndarray, points: Tuple[float, float, float, float]) -> np.ndarray:
    """Return a trapezoidal membership function with optional shoulders."""
    a, b, c, d = points
    result = np.zeros_like(universe, dtype=float)
    if b > a:
        rising = (universe >= a) & (universe <= b)
        result[rising] = (universe[rising] - a) / (b - a)
    else:
        result[universe <= c] = 1.0
    plateau = (universe >= b) & (universe <= c)
    result[plateau] = 1.0
    if d > c:
        falling = (universe >= c) & (universe <= d)
        result[falling] = np.minimum(result[falling], (d - universe[falling]) / (d - c))
    else:
        result[universe >= b] = np.minimum(result[universe >= b], 1.0)
    return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class Rule:
    lux_set: str
    traffic_set: str
    output_set: str

    @property
    def label(self) -> str:
        return f"IF ambient light is {self.lux_set} AND traffic is {self.traffic_set} THEN brightness is {self.output_set}"


class FuzzyStreetlightController:
    """Mamdani fuzzy inference system for simulated streetlights."""

    rules = (
        Rule("dark", "low", "dim"),
        Rule("dark", "medium", "bright"),
        Rule("dark", "high", "full"),
        Rule("moderate", "low", "dim"),
        Rule("moderate", "medium", "medium"),
        Rule("moderate", "high", "full"),
        Rule("bright", "low", "off"),
        Rule("bright", "medium", "off"),
        Rule("bright", "high", "dim"),
    )

    def __init__(self) -> None:
        self.lux_membership = {
            "dark": fuzz.trapmf(LUX_UNIVERSE, [0, 0, 150, 350]),
            "moderate": fuzz.trimf(LUX_UNIVERSE, [200, 500, 800]),
            "bright": fuzz.trapmf(LUX_UNIVERSE, [650, 850, 1000, 1000]),
        }
        self.traffic_membership = {
            "low": fuzz.trapmf(TRAFFIC_UNIVERSE, [0, 0, 20, 40]),
            "medium": fuzz.trimf(TRAFFIC_UNIVERSE, [25, 55, 80]),
            "high": fuzz.trapmf(TRAFFIC_UNIVERSE, [65, 85, 100, 100]),
        }
        self.brightness_membership = {
            "off": fuzz.trapmf(BRIGHTNESS_UNIVERSE, [0, 0, 5, 15]),
            "dim": fuzz.trimf(BRIGHTNESS_UNIVERSE, [10, 30, 50]),
            "medium": fuzz.trimf(BRIGHTNESS_UNIVERSE, [35, 55, 75]),
            "bright": fuzz.trimf(BRIGHTNESS_UNIVERSE, [60, 78, 92]),
            "full": fuzz.trapmf(BRIGHTNESS_UNIVERSE, [85, 95, 100, 100]),
        }

    @staticmethod
    def _membership_at(universe: np.ndarray, curve: np.ndarray, value: float) -> float:
        return float(np.interp(value, universe, curve))

    def evaluate(self, lux: float, traffic: float) -> Tuple[float, Dict[str, float]]:
        """Evaluate inputs and return brightness percentage plus rule strengths."""
        lux = float(np.clip(lux, 0, 1000))
        traffic = float(np.clip(traffic, 0, 100))
        lux_values = {name: self._membership_at(LUX_UNIVERSE, curve, lux) for name, curve in self.lux_membership.items()}
        traffic_values = {name: self._membership_at(TRAFFIC_UNIVERSE, curve, traffic) for name, curve in self.traffic_membership.items()}

        strengths: Dict[str, float] = {}
        aggregate = np.zeros_like(BRIGHTNESS_UNIVERSE)
        for index, rule in enumerate(self.rules, start=1):
            strength = min(lux_values[rule.lux_set], traffic_values[rule.traffic_set])
            strengths[f"Rule {index}: {rule.label}"] = strength
            aggregate = np.maximum(aggregate, np.minimum(strength, self.brightness_membership[rule.output_set]))

        denominator = float(np.sum(aggregate))
        brightness = float(np.sum(BRIGHTNESS_UNIVERSE * aggregate) / denominator) if denominator else 0.0
        return float(np.clip(brightness, 0, 100)), strengths

    def membership_data(self) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        return {
            "Ambient light (lux)": {key: (LUX_UNIVERSE, value) for key, value in self.lux_membership.items()},
            "Traffic density (%)": {key: (TRAFFIC_UNIVERSE, value) for key, value in self.traffic_membership.items()},
            "Brightness (%)": {key: (BRIGHTNESS_UNIVERSE, value) for key, value in self.brightness_membership.items()},
        }


def threshold_controller(lux: float, traffic: float) -> float:
    """Simple fixed-level baseline used for the comparative analysis."""
    if lux >= 800:
        return 20.0 if traffic >= 65 else 0.0
    if lux >= 350:
        return 70.0 if traffic >= 65 else 50.0
    return 100.0 if traffic >= 65 else 50.0


def scenario_rows(controller: FuzzyStreetlightController) -> Iterable[dict]:
    scenarios = ((50, 10), (50, 90), (500, 50), (950, 10), (950, 90))
    for number, (lux, traffic) in enumerate(scenarios, start=1):
        fuzzy, _ = controller.evaluate(lux, traffic)
        yield {"Test": number, "Ambient light (lux)": lux, "Traffic (%)": traffic, "Fuzzy brightness (%)": round(fuzzy, 2), "Threshold (%)": threshold_controller(lux, traffic)}


def run_dashboard() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    controller = FuzzyStreetlightController()
    st.set_page_config(page_title="Smart Streetlight Fuzzy Controller", page_icon="💡", layout="wide")
    st.title("💡 Smart Streetlight Controller Using Fuzzy Logic")
    st.caption("A simulation of Mamdani fuzzy inference using ambient light and traffic density.")

    with st.sidebar:
        st.header("Simulation inputs")
        lux = st.slider("Ambient light (lux)", 0, 1000, 250, 10)
        traffic = st.slider("Traffic density (%)", 0, 100, 50, 1)
        rated_power = st.number_input("Rated lamp power (W)", min_value=1.0, value=100.0, step=1.0)

    brightness, strengths = controller.evaluate(lux, traffic)
    estimated_power = rated_power * brightness / 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended brightness", f"{brightness:.1f}%")
    c2.metric("Estimated power", f"{estimated_power:.1f} W")
    c3.metric("Inputs", f"{lux} lux / {traffic}% traffic")
    st.progress(int(round(brightness)), text="Controller output")

    left, right = st.columns(2)
    with left:
        st.subheader("Membership functions")
        fig, axes = plt.subplots(3, 1, figsize=(8, 9))
        for axis, (title, curves) in zip(axes, controller.membership_data().items()):
            for name, (universe, curve) in curves.items():
                axis.plot(universe, curve, label=name.title())
            axis.set_title(title)
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.25)
            axis.legend(ncol=3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Active rule strengths")
        active = [(label, value) for label, value in strengths.items() if value > 0]
        if active:
            for label, value in active:
                st.write(f"**{value:.2f}** — {label}")
        else:
            st.info("No rule fired; the output is 0%.")
        st.subheader("Fuzzy response: traffic sweep")
        traffic_values = np.arange(0, 101, 1)
        response = [controller.evaluate(lux, value)[0] for value in traffic_values]
        fig2, axis = plt.subplots(figsize=(8, 4))
        axis.plot(traffic_values, response, color="#1565c0", linewidth=2)
        axis.scatter([traffic], [brightness], color="#d32f2f", zorder=3, label="Current input")
        axis.set(xlabel="Traffic density (%)", ylabel="Brightness (%)", ylim=(0, 100))
        axis.grid(alpha=0.25)
        axis.legend()
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Scenario verification")
    st.dataframe(list(scenario_rows(controller)), use_container_width=True, hide_index=True)
    st.info("This is a simulation. Estimated power assumes proportional dimming and is not a measurement of a real LED driver.")


if __name__ == "__main__":
    run_dashboard()
