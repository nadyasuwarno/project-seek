"""The behavioral signature schema (PRD section 5: candidate feature families).

A signature is a small, fixed vocabulary of measurable dynamics, grouped into
dimensions that are compared separately (R4: no single unexplained score).

Scale handling (PRD open question 3): features are marked `scale_free` when
they carry no units (ratios, exponents, 0-1 indices). By default Seeker
compares systems only on scale-free features, so a bacterium and a fish school
can be compared on *how* they move rather than how fast in absolute terms.
Absolute features (speed in um/s, period in s) are always shown as context and
can be switched into the comparison.

Similarity per feature: sim = exp(-|t(a) - t(b)| / tolerance), where t is the
identity or log10. Tolerance is the difference that drops similarity to 0.37,
so every number in the UI can be traced to one readable rule.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    family: str
    unit: str
    transform: str      # "linear" or "log"
    tolerance: float    # difference (in transformed space) at which similarity = 0.37
    scale_free: bool
    description: str


FAMILIES: dict[str, dict[str, str]] = {
    "movement": {
        "label": "Movement",
        "description": "How fast units move and how steady that speed is.",
    },
    "trajectory": {
        "label": "Trajectory shape",
        "description": "Whether paths are straight, turning, diffusive, directed, or confined.",
    },
    "collective": {
        "label": "Collective organization",
        "description": "Whether units align with each other and how they spread through space.",
    },
    "temporal": {
        "label": "Periodicity",
        "description": "Whether individual motion repeats on a regular rhythm.",
    },
    "perturbation": {
        "label": "Perturbation response",
        "description": "Baseline -> perturbation -> response -> recovery (needs a perturbation time).",
    },
}

FEATURES: list[Feature] = [
    Feature("mean_speed", "Mean speed", "movement", "um/s", "log", 1.0, False,
            "Average speed of each tracked unit, averaged over units."),
    Feature("speed_cv", "Speed variability", "movement", "CV", "linear", 0.35, True,
            "Coefficient of variation of each unit's step speeds: 0 = steady, higher = bursty."),
    Feature("straightness", "Straightness", "trajectory", "0-1", "linear", 0.2, True,
            "Net displacement / path length over 16-frame windows: 1 = straight line, ~0 = wandering."),
    Feature("turning", "Turning intensity", "trajectory", "0-1", "linear", 0.2, True,
            "Mean absolute turning angle between steps, divided by 180 degrees."),
    Feature("msd_alpha", "MSD exponent (alpha)", "trajectory", "-", "linear", 0.35, True,
            "Slope of mean squared displacement vs lag: <1 confined, ~1 diffusive, ~2 directed."),
    Feature("alignment", "Alignment (N-corrected)", "collective", "0-1", "linear", 0.2, True,
            "How aligned movement directions are in each frame, corrected for the chance "
            "alignment expected from N random units."),
    Feature("clustering", "Clark-Evans ratio", "collective", "-", "linear", 0.3, True,
            "Mean nearest-neighbour distance / expectation for random placement: "
            "<1 clustered, ~1 random, >1 evenly spaced."),
    Feature("periodicity_strength", "Periodicity strength", "temporal", "0-1", "linear", 0.2, True,
            "Share of speed-fluctuation power in one frequency band (0 = no rhythm)."),
    Feature("dominant_period", "Dominant period", "temporal", "s", "log", 0.5, False,
            "Period of the strongest rhythm in each unit's speed."),
    Feature("response_ratio", "Response ratio", "perturbation", "x baseline", "log", 0.3, True,
            "Population speed just after the perturbation divided by baseline speed."),
    Feature("recovery_time", "Recovery time", "perturbation", "s", "log", 0.5, False,
            "Time after the perturbation until speed returns within 20% of baseline."),
]

FEATURES_BY_KEY = {f.key: f for f in FEATURES}


def features_in(family: str) -> list[Feature]:
    return [f for f in FEATURES if f.family == family]
