# Smart Streetlight Controller Using Fuzzy Logic

This project simulates an intelligent streetlight using a Mamdani fuzzy
inference system. It uses ambient light (0–1000 lux) and traffic density
(0–100%) to calculate a smooth brightness recommendation (0–100%). No
hardware or sensors are required.

## Features

- Nine interpretable IF–THEN rules for dark, moderate, and bright conditions.
- NumPy implementation of fuzzification, min rule evaluation, max aggregation,
  and centroid defuzzification.
- Streamlit dashboard with sliders, power estimate, active-rule explanation,
  membership-function plots, traffic response curve, and scenario table.
- A fixed-threshold baseline for comparison.
- Automated tests for range safety, monotonic behavior, and all sample cases.

## Setup

From this directory, create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Run the dashboard

```bash
streamlit run app.py
```

Open the displayed URL, normally `http://localhost:8501`.

## Run tests

```bash
python -m pytest -q
```

## Sample scenarios

The dashboard includes these five verification cases:

| Ambient light | Traffic | Expected qualitative behavior |
|---:|---:|---|
| 50 lux | 10% | Low/dim lighting |
| 50 lux | 90% | High/full lighting |
| 500 lux | 50% | Intermediate lighting |
| 950 lux | 10% | Off or minimum lighting |
| 950 lux | 90% | Dim lighting |

Exact percentages are calculated by centroid defuzzification and are shown in
the dashboard and tests. The power figure is a simplified simulation:
`estimated_power = rated_power * brightness / 100`.

## Scope and limitations

Inputs are simulated, not measured by physical sensors. The power estimate is
not a guarantee of real driver consumption. A future version could add vehicle
detection, live ambient-light data, multiple lamps, or a measured energy study.

## Suggested report chapters

1. Introduction and problem statement
2. Literature review
3. System analysis and architecture
4. Mamdani methodology and membership functions
5. Python implementation and dashboard
6. Testing, results, and threshold comparison
7. Conclusion, limitations, and future scope
