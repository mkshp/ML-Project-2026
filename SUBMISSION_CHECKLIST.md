# SPTA Temperature Signal Project — Run and Submission Checklist

## A. Before running
- [ ] Extract the project ZIP into a normal folder (do not run it from inside the ZIP preview).
- [ ] Install Python 3.10 or later.
- [ ] Open the extracted `SPTA_Temperature_Signal_Project` folder in VS Code, Spyder, or Command Prompt.

## B. Install and run (Windows)
Open a terminal in the project folder and run:

```powershell
py -m pip install -r requirements.txt
py spta_temperature_model.py
```

If `py` is not recognized, try:

```powershell
python -m pip install -r requirements.txt
python spta_temperature_model.py
```

## C. Check the generated outputs
After the script finishes, check that these files exist:
- `data/temperature_signal_data.csv` — 336 hourly samples and all three models/noise columns.
- `results/temperature_statistics.csv` — mean, population variance and standard deviation for each temperature model.
- `results/noise_statistics.csv` — statistics for white noise and correlated noise separately.
- `results/temperature_autocorrelation.csv` — ACF values from lag 0 to lag 48.
- `results/selected_autocorrelation_lags.csv` — selected ACF values at lags 0, 1, 6, 12, 24, 36 and 48.
- `figures/temperature_signal_models.png` — visual comparison of the three signals.
- `figures/temperature_signal_acf.png` — autocorrelation of the full temperature signals.
- `figures/temperature_noise_acf.png` — autocorrelation of the two noise processes and theoretical AR(1) ACF.

Expected sample count: **336** (14 days × 24 hourly samples).

## D. Personalise the report
- [ ] Open `SPTA_Temperature_Signal_Project_Report.docx` in Word.
- [ ] Fill in your name, roll number, course/section and submission date on the cover page.
- [ ] Confirm the instructor's required file format and naming convention.
- [ ] Do not describe the CSV values as actual station measurements: this version uses a reproducible simulated dataset.

## E. What to submit
At minimum, submit the items your instructor requested. A sensible complete submission package is:
1. `SPTA_Temperature_Signal_Project_Report.pdf` (or the DOCX if editable format is requested).
2. `spta_temperature_model.py` (source code).
3. `data/temperature_signal_data.csv` (dataset).
4. `results/` (statistical and autocorrelation output CSV files).
5. `figures/` (three generated plots).
6. `README.md` and `requirements.txt` (run instructions and dependencies).

## F. Prepare for the viva
- [ ] Be able to point to the deterministic, white-noise and correlated-noise curves in Figure 1.
- [ ] Explain why white noise is independent from one time step to the next.
- [ ] Explain why AR(1) noise is correlated and what rho = 0.85 means.
- [ ] Explain mean, variance, autocorrelation and the meaning of lag 1 versus lag 24.
- [ ] Explain why the correlated-noise model looks most realistic among these simplified models.
- [ ] State honestly that this version uses simulated data and describe how real sensor/weather-station data could replace it.
