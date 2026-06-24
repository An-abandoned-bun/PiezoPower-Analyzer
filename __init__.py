"""Data analysis tools for PiezoPower Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .energy_calculator import capacitor_energy_joules, estimate_total_energy_from_steps


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for step detection and energy estimation."""

    threshold_volts: float = 1.2
    min_gap_seconds: float = 0.35
    capacitance_farads: float = 0.001


@dataclass(frozen=True)
class StepEvent:
    """A detected footstep voltage spike."""

    step_number: int
    time_s: float
    peak_voltage_v: float
    energy_j: float


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with standard columns: time_s and voltage_v.

    Accepted input column examples:
    - time_s, seconds, timestamp, timestamp_s, time
    - voltage_v, voltage, volts, adc_voltage
    """
    if data.empty:
        raise ValueError("CSV file is empty")

    lookup = {col.lower().strip(): col for col in data.columns}

    time_candidates = ["time_s", "seconds", "timestamp", "timestamp_s", "time"]
    voltage_candidates = ["voltage_v", "voltage", "volts", "adc_voltage", "sensor_voltage"]

    time_col = next((lookup[c] for c in time_candidates if c in lookup), None)
    voltage_col = next((lookup[c] for c in voltage_candidates if c in lookup), None)

    if time_col is None or voltage_col is None:
        raise ValueError(
            "CSV must contain a time column and a voltage column. "
            "Example headers: time_s, voltage_v"
        )

    result = data[[time_col, voltage_col]].copy()
    result.columns = ["time_s", "voltage_v"]
    result["time_s"] = pd.to_numeric(result["time_s"], errors="coerce")
    result["voltage_v"] = pd.to_numeric(result["voltage_v"], errors="coerce")
    result = result.dropna().sort_values("time_s").reset_index(drop=True)

    if result.empty:
        raise ValueError("CSV has no valid numeric time/voltage rows")

    return result


def detect_steps(data: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Detect footstep peaks using a threshold and minimum time gap.

    The algorithm finds local maxima above the threshold and ignores peaks that
    occur too close to the previous accepted peak.
    """
    df = normalize_columns(data)
    voltage = df["voltage_v"].to_numpy()
    time_s = df["time_s"].to_numpy()

    if len(df) < 3:
        return pd.DataFrame(columns=["step_number", "time_s", "peak_voltage_v", "energy_j"])

    candidate_indices: list[int] = []
    for idx in range(1, len(voltage) - 1):
        is_local_peak = voltage[idx] >= voltage[idx - 1] and voltage[idx] >= voltage[idx + 1]
        is_above_threshold = voltage[idx] >= config.threshold_volts
        if is_local_peak and is_above_threshold:
            candidate_indices.append(idx)

    accepted: list[int] = []
    last_time = -np.inf
    for idx in candidate_indices:
        current_time = float(time_s[idx])
        if current_time - last_time >= config.min_gap_seconds:
            accepted.append(idx)
            last_time = current_time
        elif accepted and voltage[idx] > voltage[accepted[-1]]:
            # If a higher peak occurs inside the same step window, keep the higher one.
            accepted[-1] = idx
            last_time = current_time

    events: list[StepEvent] = []
    for step_number, idx in enumerate(accepted, start=1):
        peak_voltage = float(voltage[idx])
        events.append(
            StepEvent(
                step_number=step_number,
                time_s=float(time_s[idx]),
                peak_voltage_v=peak_voltage,
                energy_j=capacitor_energy_joules(config.capacitance_farads, peak_voltage),
            )
        )

    return pd.DataFrame([event.__dict__ for event in events])


def summarize_run(data: pd.DataFrame, steps: pd.DataFrame, config: AnalysisConfig) -> dict[str, float]:
    """Create a compact experiment summary."""
    df = normalize_columns(data)
    peak_voltages: Iterable[float] = [] if steps.empty else steps["peak_voltage_v"].tolist()
    total_energy = estimate_total_energy_from_steps(list(peak_voltages), config.capacitance_farads)

    duration_s = float(df["time_s"].max() - df["time_s"].min()) if len(df) > 1 else 0.0
    return {
        "duration_s": duration_s,
        "samples": float(len(df)),
        "steps_detected": float(0 if steps.empty else len(steps)),
        "max_voltage_v": float(df["voltage_v"].max()),
        "average_voltage_v": float(df["voltage_v"].mean()),
        "peak_step_voltage_v": float(0 if steps.empty else steps["peak_voltage_v"].max()),
        "total_energy_j": float(total_energy),
        "total_energy_mj": float(total_energy * 1000),
        "steps_per_minute": float((len(steps) / duration_s) * 60) if duration_s > 0 else 0.0,
    }


def experiment_quality_score(summary: dict[str, float]) -> tuple[int, str]:
    """Return a simple quality score and interpretation for demo purposes."""
    steps = summary.get("steps_detected", 0)
    peak_v = summary.get("peak_step_voltage_v", 0)
    avg_v = summary.get("average_voltage_v", 0)

    score = 0
    score += min(int(steps * 3), 45)
    score += min(int(peak_v * 10), 35)
    score += 20 if avg_v < peak_v / 2 and peak_v > 0 else 8
    score = max(0, min(score, 100))

    if score >= 80:
        label = "Strong signal quality"
    elif score >= 55:
        label = "Usable experimental signal"
    else:
        label = "Needs better sensor contact or threshold tuning"
    return score, label
