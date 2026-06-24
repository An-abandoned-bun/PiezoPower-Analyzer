"""Electrical energy calculations for piezoelectric tile experiments."""

from __future__ import annotations


def capacitor_energy_joules(capacitance_farads: float, voltage_volts: float) -> float:
    """Return capacitor energy in joules using E = 1/2 * C * V^2.

    Args:
        capacitance_farads: Capacitor value in farads.
        voltage_volts: Voltage across the capacitor in volts.

    Raises:
        ValueError: If capacitance or voltage is negative.
    """
    if capacitance_farads < 0:
        raise ValueError("capacitance_farads cannot be negative")
    if voltage_volts < 0:
        raise ValueError("voltage_volts cannot be negative")
    return 0.5 * capacitance_farads * (voltage_volts**2)


def joules_to_millijoules(joules: float) -> float:
    """Convert joules to millijoules."""
    return joules * 1000


def joules_to_mwh(joules: float) -> float:
    """Convert joules to milliwatt-hours."""
    return joules / 3.6


def estimate_total_energy_from_steps(
    peak_voltages: list[float], capacitance_farads: float
) -> float:
    """Estimate total energy by summing capacitor energy for each detected step.

    This is a simplified experiment estimate. In real hardware, rectifier losses,
    capacitor leakage, discharge timing, and ADC calibration should be considered.
    """
    return sum(capacitor_energy_joules(capacitance_farads, max(v, 0)) for v in peak_voltages)
