"""SPTA Lab Project: Ambient Temperature Signal Modelling (complete, single file).

One file, two ways to run it:

  1) Analysis + saved outputs (CSV files and matplotlib figures):
         py spta_complete_project.py
     All three figures also pop up on screen (close the windows to finish).
     Add --no-show to only save them without opening windows.

  2) Interactive Streamlit dashboard (Plotly charts):
         py -m streamlit run spta_complete_project.py
     or simply:
         py spta_complete_project.py --dashboard

Install once (or add --install to any command above to do it automatically):
         py -m pip install -r requirements.txt
Required libraries: numpy, pandas, matplotlib (analysis + figures);
                    streamlit, plotly (dashboard only).

Models
  - Deterministic : idealised repeating 24 h temperature cycle (20 -> 30 C).
  - White noise   : deterministic + independent Gaussian noise.
  - Correlated    : deterministic + AR(1) noise c[t] = rho*c[t-1] + e[t].

The data are a reproducible simulation, NOT weather-station measurements.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

# ------------------------------ Required libraries --------------------------
# module name -> pip package.  "core" is needed for the analysis + figures;
# "dashboard" is only needed for the interactive Streamlit app.
CORE_LIBS = {"numpy": "numpy>=1.24", "pandas": "pandas>=2.0",
             "matplotlib": "matplotlib>=3.7"}
DASHBOARD_LIBS = {"streamlit": "streamlit>=1.30", "plotly": "plotly>=5.18"}


def ensure_libraries(libs: dict) -> None:
    """Stop with a clear message (or install with --install) if libs are missing."""
    missing = [pkg for mod, pkg in libs.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    if "--install" in sys.argv:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        return
    sys.exit("Missing libraries: " + ", ".join(missing) +
             "\nInstall them with:  py -m pip install -r requirements.txt"
             "\n(or run this file again with --install)")


ensure_libraries(CORE_LIBS)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# ----------------------------- Default settings -----------------------------
DEFAULTS = dict(
    seed=42, days=14, t_min=20.0, t_max=30.0, h_min=5, h_max=15,
    sigma_w=0.9, rho=0.85, sigma_c=1.2, max_lag=48,
)
SELECTED_LAGS = [0, 1, 6, 12, 24, 36, 48]


# ------------------------------ Core functions ------------------------------
def deterministic_temperature(hours, t_min=20.0, t_max=30.0, h_min=5, h_max=15):
    """Asymmetric repeating cycle: warming h_min->h_max, cooling h_max->h_min+24.

    Piecewise cosines keep the curve smooth while making warming (10 h) and
    cooling (14 h) last different amounts of time.
    """
    h = np.mod(np.asarray(hours, dtype=float), 24.0)
    mean_level = (t_min + t_max) / 2.0
    amp = (t_max - t_min) / 2.0
    out = np.empty_like(h)
    warming = (h >= h_min) & (h < h_max)
    cooling = ~warming

    out[warming] = mean_level - amp * np.cos(
        np.pi * (h[warming] - h_min) / (h_max - h_min))

    ext = h[cooling].copy()
    ext[ext < h_min] += 24.0
    cool_len = 24.0 - h_max + h_min
    out[cooling] = mean_level + amp * np.cos(np.pi * (ext - h_max) / cool_len)
    return out


def autocorrelation(values, max_lag):
    """Normalised sample ACF, lags 0..max_lag (biased estimator, r[0] = 1).

    r[k] = sum((x[t]-mean)(x[t+k]-mean)) / sum((x[t]-mean)^2)
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("values must be a 1-D array with at least two samples")
    max_lag = min(max_lag, x.size - 1)
    c = x - x.mean()
    denom = float(np.dot(c, c))
    if np.isclose(denom, 0.0):
        raise ValueError("ACF is undefined for a constant signal")
    return np.array([np.dot(c[: x.size - k], c[k:]) / denom
                     for k in range(max_lag + 1)])


def summarize(values):
    """Return (mean, population variance, standard deviation); ddof = 0."""
    return (float(np.mean(values)),
            float(np.var(values, ddof=0)),
            float(np.std(values, ddof=0)))


def simulate(seed=42, days=14, t_min=20.0, t_max=30.0, h_min=5, h_max=15,
             sigma_w=0.9, rho=0.85, sigma_c=1.2) -> pd.DataFrame:
    """Generate the three temperature models and both noise processes."""
    n = days * 24
    hours = np.arange(n, dtype=float)
    det = deterministic_temperature(hours, t_min, t_max, h_min, h_max)

    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, sigma_w, size=n)

    # AR(1): innovation SD chosen so the stationary SD of c[t] equals sigma_c.
    innov_sd = sigma_c * np.sqrt(1.0 - rho ** 2)
    corr = np.empty(n)
    corr[0] = rng.normal(0.0, sigma_c)
    innov = rng.normal(0.0, innov_sd, size=n - 1)
    for i in range(1, n):
        corr[i] = rho * corr[i - 1] + innov[i - 1]

    return pd.DataFrame({
        "Hour_from_start": hours.astype(int),
        "Hour_of_day": hours.astype(int) % 24,
        "Deterministic_temperature_C": det,
        "White_noise_C": white,
        "Temperature_with_white_noise_C": det + white,
        "Correlated_noise_C": corr,
        "Temperature_with_correlated_noise_C": det + corr,
    })


def split_signals(df: pd.DataFrame):
    models = {
        "Deterministic model": df["Deterministic_temperature_C"].to_numpy(),
        "White-noise model": df["Temperature_with_white_noise_C"].to_numpy(),
        "Correlated-noise model": df["Temperature_with_correlated_noise_C"].to_numpy(),
    }
    noises = {
        "White noise only": df["White_noise_C"].to_numpy(),
        "Correlated noise only": df["Correlated_noise_C"].to_numpy(),
    }
    return models, noises


# ------------------------------ Figure builders -----------------------------
# These three matplotlib figures are the Figure 1-3 of the report. They are
# generated entirely from code, so the report plots can be recreated any time.
def fig_signal_models(df, models, hours=72):
    """Figure 1: deterministic, white-noise and correlated-noise signals."""
    vis = slice(0, hours)
    x = df["Hour_from_start"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, (name, sig) in zip(axes, models.items()):
        ax.plot(x[vis], sig[vis], linewidth=1.25)
        ax.set_title(name)
        ax.set_ylabel("Temperature (deg C)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time from start (hours)")
    fig.suptitle(f"Hourly ambient-temperature signal: first {hours} hours", fontsize=14)
    fig.tight_layout()
    return fig


def fig_temperature_acf(models, max_lag):
    """Figure 2: autocorrelation of the three temperature models."""
    lags = np.arange(max_lag + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, sig in models.items():
        ax.plot(lags, autocorrelation(sig, max_lag), marker="o",
                markersize=2.5, linewidth=1.2, label=name)
    ax.axhline(0.0, linewidth=0.8)
    ax.set(title="Autocorrelation of the three temperature models",
           xlabel="Lag (hours)", ylabel="Autocorrelation", xlim=(0, max_lag))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_noise_acf(noises, rho, max_lag):
    """Figure 3: noise-only autocorrelation with the theoretical AR(1) curve."""
    lags = np.arange(max_lag + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, nz in noises.items():
        ax.plot(lags, autocorrelation(nz, max_lag), marker="o",
                markersize=2.5, linewidth=1.2, label=name)
    ax.plot(lags, rho ** lags, linestyle="--", linewidth=1.2,
            label=f"Theoretical AR(1) ACF (rho={rho})")
    ax.axhline(0.0, linewidth=0.8)
    ax.set(title="Autocorrelation of the noise processes",
           xlabel="Lag (hours)", ylabel="Autocorrelation", xlim=(0, max_lag))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def build_report_figures(df, models, noises, rho, max_lag) -> dict:
    """Return {filename: matplotlib Figure} for all three report figures."""
    return {
        "temperature_signal_models.png": fig_signal_models(df, models),
        "temperature_signal_acf.png": fig_temperature_acf(models, max_lag),
        "temperature_noise_acf.png": fig_noise_acf(noises, rho, max_lag),
    }


def fig_to_png_bytes(fig, dpi=180) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


# ------------------------------ Mode 1: analysis ----------------------------
def _can_open_windows() -> bool:
    """True if a graphical display is available for matplotlib windows."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def run_analysis(show: bool = True) -> None:
    if show and not _can_open_windows():
        print("No graphical display found: figures will be saved but not shown.")
        show = False
    if not show:
        plt.switch_backend("Agg")
    p = DEFAULTS
    data_dir, res_dir, fig_dir = ROOT / "data", ROOT / "results", ROOT / "figures"
    for folder in (data_dir, res_dir, fig_dir):
        folder.mkdir(parents=True, exist_ok=True)

    df = simulate(p["seed"], p["days"], p["t_min"], p["t_max"], p["h_min"],
                  p["h_max"], p["sigma_w"], p["rho"], p["sigma_c"])
    models, noises = split_signals(df)
    lags = np.arange(p["max_lag"] + 1)

    # Statistics tables
    stats = pd.DataFrame(
        [(n, *summarize(s)) for n, s in models.items()],
        columns=["Model", "Mean temperature (deg C)",
                 "Variance (deg C^2)", "Standard deviation (deg C)"])
    nstats = pd.DataFrame(
        [(n, *summarize(s)) for n, s in noises.items()],
        columns=["Noise process", "Mean noise (deg C)",
                 "Variance (deg C^2)", "Standard deviation (deg C)"])

    # Autocorrelation tables
    acf = pd.DataFrame({"Lag (hours)": lags})
    for n, s in models.items():
        acf[f"ACF - {n}"] = autocorrelation(s, p["max_lag"])
    for n, s in noises.items():
        acf[f"ACF - {n}"] = autocorrelation(s, p["max_lag"])
    acf["Theoretical AR(1) noise ACF"] = p["rho"] ** lags
    selected = acf[acf["Lag (hours)"].isin(SELECTED_LAGS)]

    # Save CSV outputs
    df.to_csv(data_dir / "temperature_signal_data.csv", index=False)
    stats.to_csv(res_dir / "temperature_statistics.csv", index=False)
    nstats.to_csv(res_dir / "noise_statistics.csv", index=False)
    acf.to_csv(res_dir / "temperature_autocorrelation.csv", index=False)
    selected.to_csv(res_dir / "selected_autocorrelation_lags.csv", index=False)

    # Report figures (same builders are used by the dashboard)
    for filename, fig in build_report_figures(df, models, noises, p["rho"], p["max_lag"]).items():
        fig.savefig(fig_dir / filename, dpi=180, bbox_inches="tight")
        if show:
            try:
                fig.canvas.manager.set_window_title(filename)
            except Exception:
                pass
        else:
            plt.close(fig)

    # Console report
    fmt = lambda x: f"{x:.4f}"
    print("\nSPTA - AMBIENT TEMPERATURE SIGNAL MODEL")
    print(f"Samples: {len(df)} hourly readings across {p['days']} days")
    print("\nTemperature-model statistics (population variance):")
    print(stats.to_string(index=False, float_format=fmt))
    print("\nNoise-process statistics:")
    print(nstats.to_string(index=False, float_format=fmt))
    print("\nSelected ACF values:")
    print(selected.to_string(index=False, float_format=fmt))
    print(f"\nOutputs saved under: {ROOT}")
    print("Tip: run  py spta_complete_project.py --dashboard  for the interactive app.")

    if show:
        print("\nShowing all 3 figures - close the windows to finish.")
        plt.show()  # displays Figure 1, 2 and 3 together


# ------------------------------ Mode 2: dashboard ---------------------------
def run_dashboard() -> None:
    ensure_libraries(DASHBOARD_LIBS)
    plt.switch_backend("Agg")  # Streamlit renders figures itself via st.pyplot
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="SPTA Temperature Signal Modelling", layout="wide")
    colors = {"Deterministic model": "#1f77b4",
              "White-noise model": "#ff7f0e",
              "Correlated-noise model": "#2ca02c"}

    cached_sim = st.cache_data(simulate)

    def style(fig, title, xt, yt, height=420):
        fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, height=height,
                          margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.2), hovermode="x unified")
        return fig

    # Sidebar controls
    d = DEFAULTS
    sb = st.sidebar
    sb.header("Simulation settings")
    seed = sb.number_input("Random seed", 0, 10_000, d["seed"])
    days = sb.slider("Days", 3, 60, d["days"])
    t_min = sb.slider("Daily minimum (°C)", 0.0, 30.0, d["t_min"], 0.5)
    t_max = sb.slider("Daily maximum (°C)", 10.0, 45.0, d["t_max"], 0.5)
    h_min = sb.slider("Hour of minimum", 0, 11, d["h_min"])
    h_max = sb.slider("Hour of maximum", 12, 20, d["h_max"])
    sigma_w = sb.slider("White-noise std (°C)", 0.0, 3.0, d["sigma_w"], 0.1)
    rho = sb.slider("AR(1) coefficient ρ", 0.0, 0.99, d["rho"], 0.01)
    sigma_c = sb.slider("Correlated-noise stationary std (°C)", 0.1, 3.0, d["sigma_c"], 0.1)
    max_lag = sb.slider("Max ACF lag (hours)", 12, 96, d["max_lag"], 6)

    if t_max <= t_min:
        st.error("Daily maximum must be greater than the daily minimum.")
        st.stop()

    df = cached_sim(seed, days, t_min, t_max, h_min, h_max, sigma_w, rho, sigma_c)
    models, noises = split_signals(df)
    lags = np.arange(max_lag + 1)

    # Header + metrics
    st.title("Ambient Temperature Signal Modelling")
    st.caption("SPTA Lab Project · Moksh Pandit (A25ARIU0020) · "
               "Data are a reproducible simulation, not weather-station measurements.")
    for col, (name, sig) in zip(st.columns(3), models.items()):
        m, v, _ = summarize(sig)
        col.metric(name, f"{m:.3f} °C mean", f"variance {v:.3f} °C²", delta_color="off")

    tabs = st.tabs(["Signals", "Statistics", "Autocorrelation", "Noise ACF",
                    "Report figures", "Data & download", "Your own data"])

    # --- Signals
    with tabs[0]:
        visible = st.slider("Hours to display", 24, days * 24, min(72, days * 24), 24)
        layout = st.radio("Layout", ["Overlay", "Separate panels"], horizontal=True)
        x = df["Hour_from_start"][:visible]
        if layout == "Overlay":
            fig = go.Figure()
            for name, sig in models.items():
                fig.add_scatter(x=x, y=sig[:visible], name=name, line=dict(color=colors[name]))
            st.plotly_chart(style(fig, "Hourly ambient temperature",
                                  "Time from start (hours)", "Temperature (°C)", 480),
                            use_container_width=True)
        else:
            for name, sig in models.items():
                fig = go.Figure(go.Scatter(x=x, y=sig[:visible], name=name,
                                           line=dict(color=colors[name])))
                st.plotly_chart(style(fig, name, "Time from start (hours)",
                                      "Temperature (°C)", 260), use_container_width=True)
        st.markdown(
            "- **Deterministic**: smooth and repeats exactly every 24 h.\n"
            "- **White noise**: independent hourly jumps around the same daily shape.\n"
            "- **Correlated noise**: deviations persist for several hours (thermal inertia).")

    # --- Statistics
    with tabs[1]:
        stats = pd.DataFrame([(n, *summarize(s)) for n, s in models.items()],
                             columns=["Model", "Mean (°C)", "Variance (°C²)", "Std. dev. (°C)"])
        nstats = pd.DataFrame([(n, *summarize(s)) for n, s in noises.items()],
                              columns=["Noise process", "Mean (°C)", "Variance (°C²)",
                                       "Std. dev. (°C)"])
        c1, c2 = st.columns(2)
        c1.subheader("Temperature models")
        c1.dataframe(stats.style.format({c: "{:.3f}" for c in stats.columns[1:]}),
                     hide_index=True, use_container_width=True)
        c2.subheader("Noise only")
        c2.dataframe(nstats.style.format({c: "{:.3f}" for c in nstats.columns[1:]}),
                     hide_index=True, use_container_width=True)

        fig = go.Figure()
        for name, sig in models.items():
            fig.add_histogram(x=sig, name=name, opacity=0.55, nbinsx=30,
                              marker_color=colors[name])
        fig.update_layout(barmode="overlay")
        st.plotly_chart(style(fig, "Distribution of temperature values",
                              "Temperature (°C)", "Count"), use_container_width=True)

        cols = ["Deterministic_temperature_C", "Temperature_with_white_noise_C",
                "Temperature_with_correlated_noise_C"]
        profile = df.groupby("Hour_of_day")[cols].mean()
        fig = go.Figure()
        for name, col in zip(models, cols):
            fig.add_scatter(x=profile.index, y=profile[col], name=name,
                            line=dict(color=colors[name]))
        st.plotly_chart(style(fig, "Average temperature by hour of day",
                              "Hour of day", "Temperature (°C)"), use_container_width=True)

    # --- Autocorrelation
    with tabs[2]:
        fig = go.Figure()
        acf_table = pd.DataFrame({"Lag (hours)": lags})
        for name, sig in models.items():
            r = autocorrelation(sig, max_lag)
            acf_table[name] = r
            fig.add_scatter(x=lags, y=r, mode="lines+markers", name=name,
                            line=dict(color=colors[name]), marker=dict(size=4))
        fig.add_hline(y=0, line_width=1, line_color="gray")
        st.plotly_chart(style(fig, "Autocorrelation of the three temperature models",
                              "Lag (hours)", "Autocorrelation"), use_container_width=True)
        st.subheader("Selected lags")
        pick = [k for k in SELECTED_LAGS if k <= max_lag]
        st.dataframe(acf_table[acf_table["Lag (hours)"].isin(pick)]
                     .style.format({c: "{:.3f}" for c in acf_table.columns[1:]}),
                     hide_index=True, use_container_width=True)
        st.info("The ACF peaks near lag 24 and dips near lag 12 because of the "
                "repeating daily cycle.")

    # --- Noise ACF
    with tabs[3]:
        fig = go.Figure()
        for name, nz in noises.items():
            fig.add_scatter(x=lags, y=autocorrelation(nz, max_lag), mode="lines+markers",
                            name=name, marker=dict(size=4))
        fig.add_scatter(x=lags, y=rho ** lags, name=f"Theoretical AR(1): ρ^k (ρ={rho})",
                        line=dict(dash="dash", color="gray"))
        fig.add_hline(y=0, line_width=1, line_color="gray")
        st.plotly_chart(style(fig, "Autocorrelation of the noise processes",
                              "Lag (hours)", "Autocorrelation"), use_container_width=True)
        st.markdown(
            f"White noise is ≈ 0 at every non-zero lag. AR(1) noise follows ρ^k "
            f"(lag 1 ≈ {rho:.2f}, lag 5 ≈ {rho ** 5:.2f}), but an estimate from "
            f"{len(df)} samples will not match the theory exactly.")

    # --- Report figures (matplotlib, identical to Figures 1-3 of the report)
    with tabs[4]:
        st.write("The three matplotlib figures used in the project report, "
                 "regenerated live from the current sidebar settings.")
        captions = {
            "temperature_signal_models.png": "Figure 1. First 72 hours of the three temperature models.",
            "temperature_signal_acf.png": "Figure 2. ACF of the complete temperature signals.",
            "temperature_noise_acf.png": "Figure 3. ACF of the noise processes vs theoretical AR(1).",
        }
        for filename, fig in build_report_figures(df, models, noises, rho, max_lag).items():
            png = fig_to_png_bytes(fig)
            st.pyplot(fig, clear_figure=True)
            st.caption(captions[filename])
            st.download_button(f"Download {filename}", png, filename, "image/png",
                               key=f"dl_{filename}")
            plt.close(fig)

    # --- Data & download
    with tabs[5]:
        st.dataframe(df, use_container_width=True, height=360)
        st.download_button("Download data CSV", df.to_csv(index=False).encode(),
                           "temperature_signal_data.csv", "text/csv")

    # --- Your own data
    with tabs[6]:
        st.write("Upload a CSV of real hourly temperatures to run the same statistics and ACF.")
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            real = pd.read_csv(up)
            numeric = real.select_dtypes("number").columns.tolist()
            if not numeric:
                st.error("No numeric columns found.")
            else:
                col = st.selectbox("Temperature column", numeric)
                sig = real[col].dropna().to_numpy()
                m, v, s = summarize(sig)
                a, b, c = st.columns(3)
                a.metric("Mean", f"{m:.3f}")
                b.metric("Variance", f"{v:.3f}")
                c.metric("Std. dev.", f"{s:.3f}")
                fig = go.Figure(go.Scatter(y=sig, mode="lines", name=col))
                st.plotly_chart(style(fig, col, "Sample index", col), use_container_width=True)
                lag_r = min(max_lag, len(sig) - 1)
                fig = go.Figure(go.Scatter(x=np.arange(lag_r + 1),
                                           y=autocorrelation(sig, lag_r),
                                           mode="lines+markers", name="ACF"))
                st.plotly_chart(style(fig, "Autocorrelation", "Lag", "ACF"),
                                use_container_width=True)


# ---------------------------------- Entry point -----------------------------
def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:
            return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPTA temperature signal project")
    parser.add_argument("--dashboard", action="store_true",
                        help="launch the interactive Streamlit dashboard")
    parser.add_argument("--no-show", action="store_true",
                        help="save figures without opening any windows")
    parser.add_argument("--install", action="store_true",
                        help="pip-install any missing required libraries first")
    args = parser.parse_args()
    if args.dashboard:
        ensure_libraries(DASHBOARD_LIBS)
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
    else:
        run_analysis(show=not args.no_show)
