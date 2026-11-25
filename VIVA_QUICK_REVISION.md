# SPTA Viva — Quick Revision

## Project in one sentence
I modelled hourly outdoor temperature using a repeating daily temperature curve, then added independent white noise and temporally correlated AR(1) noise, and compared their mean, variance and autocorrelation.

## Three models
1. **Deterministic:** the same daily temperature curve repeats every 24 hours. It has no random disturbance.
2. **White-noise model:** deterministic temperature plus independent, zero-mean Gaussian noise. This can cause abrupt hour-to-hour changes.
3. **Correlated-noise model:** deterministic temperature plus AR(1) noise. A disturbance partly carries over from the previous hour, so deviations persist more smoothly.

## Main equations
- White-noise model: `T_white[n] = T_d[n] + w[n]`, where `w[n]` is independent Gaussian noise.
- Correlated noise: `c[n] = rho*c[n-1] + e[n]`, with `rho = 0.85` and independent innovation `e[n]`.
- Correlated-noise temperature: `T_corr[n] = T_d[n] + c[n]`.
- Mean: `mean(T) = sum(T[n]) / N`.
- Population variance: `var(T) = sum((T[n] - mean(T))**2) / N`.
- Autocorrelation at lag k: correlation of the signal with a copy shifted by k samples. Here, one sample is one hour.

## Results from this fixed simulation
| Model | Mean (°C) | Population variance (°C²) |
|---|---:|---:|
| Deterministic | 25.000 | 12.500 |
| White noise added | 24.987 | 13.238 |
| Correlated noise added | 24.777 | 14.225 |

The small differences in means are caused by the finite random noise samples. Noise is zero-mean in expectation, not necessarily exactly zero in one finite simulation.

## Why does correlated noise look most realistic?
Outdoor temperature has continuity and thermal inertia. Air, roads, buildings and soil do not instantly change temperature independently every hour. Weather effects such as cloud cover and wind can also persist. Therefore, temperature deviations often last for multiple hours. AR(1) noise represents this persistence. It is still a simplified model, not a complete weather forecast model.

## Common viva questions
**What is a signal?** A quantity that changes with an independent variable; here temperature changes with time.

**What is white noise?** A random process with zero expected mean, constant variance and no correlation between different time samples.

**What is correlated noise?** Noise in which values at nearby times are statistically related.

**What does rho = 0.85 mean?** A positive AR(1) coefficient of 0.85 means much of the previous hour's disturbance carries into the current hour; the influence decays over later hours.

**What does mean show?** The average temperature level over the observation interval.

**What does variance show?** How widely the temperature samples spread around their mean.

**What is autocorrelation?** Similarity between a signal and a time-shifted version of itself.

**What is lag 1?** Comparing each hour with the next hour.

**What is lag 24?** Comparing readings separated by 24 hours. A daily cycle can produce high autocorrelation around this lag.

**Why is the full-signal ACF different from the noise-only ACF?** The full temperature signal includes the repeating daily cycle, while noise-only ACF isolates the random component.

**Is the dataset actual weather-station data?** No. It is a fixed-seed simulation used to demonstrate the required signal-processing concepts. Real hourly weather observations could be substituted in a future version.
