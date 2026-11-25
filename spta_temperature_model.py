"""SPTA Lab Task: Modelling an ambient-temperature signal.

This reproducible simulation creates an idealized daily temperature cycle,
adds independent white noise and correlated AR(1) noise, and computes
mean, population variance and sample autocorrelation.

The generated values are simulated coursework data, not measurements from a
weather station. See README.md and the accompanying report for assumptions.

Install:  py -m pip install -r requirements.txt
Run:      py spta_temperature_model.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for folder in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------- Simulation settings ---------------------------
RANDOM_SEED = 42
DAYS = 14
SAMPLES_PER_DAY = 24
N = DAYS * SAMPLES_PER_DAY
HOURS = np.arange(N, dtype=float)

MIN_TEMPERATURE_C = 20.0
MAX_TEMPERATURE_C = 30.0
MEAN_LEVEL_C = (MIN_TEMPERATURE_C + MAX_TEMPERATURE_C) / 2.0
DAILY_MINIMUM_HOUR = 5.0       # illustrative minimum near sunrise
DAILY_MAXIMUM_HOUR = 15.0      # illustrative afternoon maximum

WHITE_NOISE_STD_C = 0.9
AR_COEFFICIENT = 0.85           # positive value => persistence between hours
CORRELATED_NOISE_STATIONARY_STD_C = 1.2
MAX_LAG_HOURS = 48


def deterministic_temperature(hour_indices: np.ndarray) -> np.ndarray:
    """Return an asymmetric, repeating 24-hour temperature cycle.

    The temperature rises from its minimum at 05:00 to its maximum at 15:00,
    then falls until 05:00 the next day. Piecewise cosine segments make the
    warming and cooling durations different while keeping the curve smooth.
    """
    h = np.mod(np.asarray(hour_indices, dtype=float), 24.0)
    result = np.empty_like(h)
    warming = (h >= DAILY_MINIMUM_HOUR) & (h < DAILY_MAXIMUM_HOUR)
    cooling = ~warming

    # Warming segment: 05:00 -> 15:00.
    result[warming] = MEAN_LEVEL_C - (MAX_TEMPERATURE_C - MIN_TEMPERATURE_C) / 2.0 * np.cos(
        np.pi * (h[warming] - DAILY_MINIMUM_HOUR)
        / (DAILY_MAXIMUM_HOUR - DAILY_MINIMUM_HOUR)
    )

    # Cooling segment: 15:00 -> 05:00 next day (use an extended hour axis).
    extended_h = h[cooling].copy()
    extended_h[extended_h < DAILY_MINIMUM_HOUR] += 24.0
    cooling_duration = 24.0 - DAILY_MAXIMUM_HOUR + DAILY_MINIMUM_HOUR
    result[cooling] = MEAN_LEVEL_C + (MAX_TEMPERATURE_C - MIN_TEMPERATURE_C) / 2.0 * np.cos(
        np.pi * (extended_h - DAILY_MAXIMUM_HOUR) / cooling_duration
    )
    return result


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized sample ACF for lags 0..max_lag.

    r[k] = sum((x[t]-mean)*(x[t+k]-mean)) / sum((x[t]-mean)^2)
    This common biased estimator guarantees r[0] = 1.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("values must be a 1-D array with at least two samples")
    if not 0 <= max_lag < x.size:
        raise ValueError("max_lag must be between 0 and len(values)-1")
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    if np.isclose(denominator, 0.0):
        raise ValueError("ACF is undefined for a constant signal")
    return np.array([
        np.dot(centered[: x.size - lag], centered[lag:]) / denominator
        for lag in range(max_lag + 1)
    ])


def summarize(values: np.ndarray) -> dict[str, float]:
    """Return mean, population variance and standard deviation."""
    return {
        "Mean": float(np.mean(values)),
        "Population variance": float(np.var(values, ddof=0)),
        "Standard deviation": float(np.std(values, ddof=0)),
    }


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Idealized deterministic temperature profile.
    deterministic = deterministic_temperature(HOURS)

    # 2) Independent, zero-mean Gaussian white noise.
    white_noise = rng.normal(0.0, WHITE_NOISE_STD_C, size=N)
    with_white_noise = deterministic + white_noise

    # 3) Correlated AR(1) noise: c[t] = rho*c[t-1] + innovation[t].
    # Set innovation SD so the stationary noise SD is approximately 1.2 C.
    innovation_std = CORRELATED_NOISE_STATIONARY_STD_C * np.sqrt(1.0 - AR_COEFFICIENT**2)
    correlated_noise = np.empty(N, dtype=float)
    correlated_noise[0] = rng.normal(0.0, CORRELATED_NOISE_STATIONARY_STD_C)
    innovations = rng.normal(0.0, innovation_std, size=N - 1)
    for i in range(1, N):
        correlated_noise[i] = AR_COEFFICIENT * correlated_noise[i - 1] + innovations[i - 1]
    with_correlated_noise = deterministic + correlated_noise

    models = {
        "Deterministic model": deterministic,
        "White-noise model": with_white_noise,
        "Correlated-noise model": with_correlated_noise,
    }
    noise_processes = {
        "White noise only": white_noise,
        "Correlated noise only": correlated_noise,
    }

    # ------------------------------ Statistics ------------------------------
    statistics = pd.DataFrame([
        {"Model": name, **summarize(signal)} for name, signal in models.items()
    ])
    statistics.columns = [
        "Model", "Mean temperature (deg C)",
        "Variance (deg C^2)", "Standard deviation (deg C)"
    ]
    statistics.to_csv(RESULTS_DIR / "temperature_statistics.csv", index=False)

    noise_statistics = pd.DataFrame([
        {"Noise process": name, **summarize(signal)}
        for name, signal in noise_processes.items()
    ])
    noise_statistics.columns = [
        "Noise process", "Mean noise (deg C)",
        "Variance (deg C^2)", "Standard deviation (deg C)"
    ]
    noise_statistics.to_csv(RESULTS_DIR / "noise_statistics.csv", index=False)

    # --------------------------- Autocorrelation -----------------------------
    lags = np.arange(MAX_LAG_HOURS + 1)
    acf_frame = pd.DataFrame({"Lag (hours)": lags})
    for name, signal in models.items():
        acf_frame[f"ACF - {name}"] = autocorrelation(signal, MAX_LAG_HOURS)
    for name, noise in noise_processes.items():
        acf_frame[f"ACF - {name}"] = autocorrelation(noise, MAX_LAG_HOURS)
    acf_frame["Theoretical AR(1) noise ACF"] = AR_COEFFICIENT ** lags
    acf_frame.to_csv(RESULTS_DIR / "temperature_autocorrelation.csv", index=False)

    selected_lags = [0, 1, 6, 12, 24, 36, 48]
    selected = acf_frame[acf_frame["Lag (hours)"].isin(selected_lags)].copy()
    selected.to_csv(RESULTS_DIR / "selected_autocorrelation_lags.csv", index=False)

    # ------------------------------- Raw data --------------------------------
    data = pd.DataFrame({
        "Hour_from_start": HOURS.astype(int),
        "Hour_of_day": (HOURS.astype(int) % 24),
        "Deterministic_temperature_C": deterministic,
        "White_noise_C": white_noise,
        "Temperature_with_white_noise_C": with_white_noise,
        "Correlated_noise_C": correlated_noise,
        "Temperature_with_correlated_noise_C": with_correlated_noise,
    })
    data.to_csv(DATA_DIR / "temperature_signal_data.csv", index=False)

    # ------------------------------- Figures ---------------------------------
    # Signal comparison: display first 72 hours so daily cycles are visible.
    visible = slice(0, 72)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (name, signal) in zip(axes, models.items()):
        ax.plot(HOURS[visible], signal[visible], linewidth=1.25)
        ax.set_title(name)
        ax.set_ylabel("Temperature (deg C)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time from start (hours)")
    fig.suptitle("Hourly ambient-temperature signal: first 72 hours", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temperature_signal_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ACF of full temperature signals (up to 48 hours).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, signal in models.items():
        ax.plot(lags, autocorrelation(signal, MAX_LAG_HOURS),
                marker="o", markersize=2.5, linewidth=1.2, label=name)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Autocorrelation of the three temperature models")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xlim(0, MAX_LAG_HOURS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temperature_signal_acf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ACF of noise alone: this isolates independence vs temporal persistence.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, noise in noise_processes.items():
        ax.plot(lags, autocorrelation(noise, MAX_LAG_HOURS),
                marker="o", markersize=2.5, linewidth=1.2, label=name)
    ax.plot(lags, AR_COEFFICIENT ** lags, linestyle="--", linewidth=1.2,
            label=f"Theoretical AR(1) ACF (rho={AR_COEFFICIENT})")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Autocorrelation of the noise processes")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xlim(0, MAX_LAG_HOURS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temperature_noise_acf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nSPTA - AMBIENT TEMPERATURE SIGNAL MODEL")
    print(f"Samples: {N} hourly readings across {DAYS} days")
    print("\nTemperature-model statistics (population variance):")
    print(statistics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nNoise-process statistics:")
    print(noise_statistics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSelected ACF values:")
    print(selected.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutputs saved under: {ROOT}")


if __name__ == "__main__":
    main()
