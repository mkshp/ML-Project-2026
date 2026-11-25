# SPTA Lab Project: Ambient Temperature Signal Modelling

## Project overview

This project models hourly outdoor ambient temperature using three signal models:

1. **Deterministic model** — an idealized daily warming and cooling cycle.
2. **White-noise model** — the deterministic signal plus independent Gaussian noise.
3. **Correlated-noise model** — the deterministic signal plus AR(1) noise, where each noise value partly depends on the previous hour.

For every temperature model, the program computes the mean, population variance, standard deviation and autocorrelation. It also calculates the autocorrelation of the noise processes separately so that white-noise independence and AR(1) persistence can be compared directly.

> **Data note:** The data are a reproducible simulation designed to resemble a plausible daily temperature pattern. They are not measured weather-station observations. This is appropriate for demonstrating the signal-processing concepts; actual sensor data can be substituted as a future extension.

## Signal assumptions

- 14 days of data, sampled once per hour (336 samples).
- Daily minimum: 20 °C at approximately 05:00.
- Daily maximum: 30 °C at approximately 15:00.
- White-noise standard deviation: 0.9 °C.
- AR(1) coefficient: 0.85.
- Correlated-noise stationary standard deviation: 1.2 °C.
- Fixed random seed: 42, so results are reproducible.

## Install and run on Windows

1. Extract the ZIP file.
2. Open the extracted folder in VS Code, or open Command Prompt in that folder.
3. Install the dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

4. Run the model:

   ```powershell
   py spta_temperature_model.py
   ```

The program creates/updates the CSV files in `data/` and `results/`, and the plots in `figures/`.

## Project structure

```text
SPTA_Temperature_Signal_Project/
├── spta_temperature_model.py
├── requirements.txt
├── README.md
├── SPTA_Temperature_Signal_Project_Report.docx
├── data/
│   └── temperature_signal_data.csv
├── results/
│   ├── temperature_statistics.csv
│   ├── noise_statistics.csv
│   ├── temperature_autocorrelation.csv
│   └── selected_autocorrelation_lags.csv
└── figures/
    ├── temperature_signal_models.png
    ├── temperature_signal_acf.png
    └── temperature_noise_acf.png
```

## Main equations

Let `t` denote the hour index and `h = t mod 24`. The deterministic component is a repeating daily cycle. It rises from 20 °C at 05:00 to 30 °C at 15:00, then cools until 05:00 the next day.

- White noise: `w[t] ~ Normal(0, sigma_w^2)` and `T_white[t] = T_d[t] + w[t]`.
- Correlated noise: `c[t] = rho*c[t-1] + e[t]`, where `e[t]` is independent Gaussian innovation; `T_corr[t] = T_d[t] + c[t]`.
- Mean: `mean(T) = sum(T[t]) / N`.
- Population variance: `var(T) = sum((T[t] - mean(T))^2) / N`.
- ACF: `r[k] = sum((T[t]-mean(T))*(T[t+k]-mean(T))) / sum((T[t]-mean(T))^2)`.

## What to explain in the viva

The correlated-noise model is the most realistic of the three simplified models. Outdoor temperature has thermal inertia: the air and surrounding surfaces do not heat or cool independently from one hour to the next. Clouds, wind and other weather effects often influence temperature over several consecutive hours. AR(1) noise represents this short-term persistence. White noise changes independently each hour and can look too jagged, while the deterministic model is overly smooth and repeats exactly every day.

The correlated-noise model is still simplified. A production weather model would use actual sensor data and additional variables such as solar radiation, cloud cover, wind, humidity and season.
