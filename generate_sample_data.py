import pandas as pd

from src.analyzer import AnalysisConfig, detect_steps, summarize_run


def test_detect_steps_simple_signal():
    data = pd.DataFrame(
        {
            "time_s": [0, 0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
            "voltage_v": [0.1, 0.2, 2.0, 0.3, 0.2, 1.8, 0.2],
        }
    )
    steps = detect_steps(data, AnalysisConfig(threshold_volts=1.0, min_gap_seconds=0.35))
    assert len(steps) == 2
    assert steps["peak_voltage_v"].max() == 2.0


def test_summarize_run():
    data = pd.DataFrame({"time_s": [0, 1, 2], "voltage_v": [0.1, 2.0, 0.1]})
    config = AnalysisConfig(threshold_volts=1.0, min_gap_seconds=0.35, capacitance_farads=0.001)
    steps = detect_steps(data, config)
    summary = summarize_run(data, steps, config)
    assert summary["steps_detected"] == 1
    assert summary["total_energy_j"] > 0
