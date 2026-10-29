# Rest-activity rhythm validation

These scripts compare Forest's circadian rhythm metrics with
independent implementations in ActCR and CosinorPy.

## Test data

Six deterministic synthetic activity patterns are used:

- Cosine rhythms peaking at 14:00 and 16:00
- Noisy cosine
- Square wave
- Square wave with simulated naps
- White noise

Each recording contains seven days of data at one-minute and
hourly resolution. The generator uses a fixed random seed.

## Reference environments

The benchmark was run with:

Forest environment:
- Python 3.12.6
- NumPy 2.4.6
- pandas 3.0.3
- SciPy 1.17.1

CosinorPy reference environment:
- Python 3.12.6
- CosinorPy 3.1
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.16.3
- seaborn 0.13.2

R reference environment:
- R 4.6
- ActCR 0.4.0

CosinorPy runs in a separate Python environment because the
tested dependency versions differ from Forest's environment.

To create that reference environment, run from the Forest root:

    python3.12 -m venv validation/rest_activity/.venv-reference

    validation/rest_activity/.venv-reference/bin/python -m pip install -r validation/rest_activity/requirements-reference.txt

Install ActCR 0.4.0 in the R environment separately and verify
the installed version before running the comparison.

## Running the benchmark

Run these commands from the Forest repository root, with
Forest's virtual environment activated.

1. Generate the synthetic recordings:

    python validation/rest_activity/make_profiles.py

2. Calculate ActCR reference metrics:

    Rscript validation/rest_activity/reference_actcr.R

3. Calculate CosinorPy reference metrics:

    validation/rest_activity/.venv-reference/bin/python validation/rest_activity/reference_cosinor.py

4. Calculate Forest metrics and compare them:

    python validation/rest_activity/compare.py

5. Review the differences:

    python validation/rest_activity/audit_validation.py

The generated profiles and CSV results are excluded from Git.

## Results

The benchmark produces 108 comparisons across nine metrics.

The reproduced results were:

- 100 agreements within the selected numerical tolerances
- 6 documented window-definition differences
- 2 separately classified white-noise acrophase estimates
- 0 unexplained discrepancies flagged by the audit

IS and IV agreed to approximately machine precision.
Cosinor mesor, amplitude, and R-squared also agreed closely.

Forest allows L5 and M10 windows to cross midnight, whereas
the ActCR reference searches within a single day. This explains
the six classified window differences.

CosinorPy estimates peak timing using a discretized fitted
curve, producing small differences from Forest's analytical
phase calculation.

The benchmark tests numerical agreement on complete synthetic
recordings. It is not a validation of smartphone wear behavior,
clinical outcomes, or all possible missing-data patterns.

The classification criteria and numerical tolerances are defined
in audit_validation.py.
