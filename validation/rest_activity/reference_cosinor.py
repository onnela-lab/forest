from pathlib import Path

import numpy as np
import pandas as pd
from CosinorPy import cosinor

validation_dir = Path(__file__).resolve().parent

results = []

for path in sorted((validation_dir / "profiles").glob("*.csv")):
    name, resolution = path.stem.rsplit("_", 1)

    data = pd.read_csv(path)
    activity = data["activity"].to_numpy(dtype=float)

    step_hours = {
        "1min": 1 / 60,
        "1h": 1.0,
    }[resolution]

    hours = np.arange(len(activity)) * step_hours

    model, _, params, _, _ = cosinor.fit_me(
        hours,
        activity,
        n_components=1,
        period=24,
        model_type="lin",
        plot=False,
    )

    peak_hour = (
        -params["acrophase"] * 24 / (2 * np.pi)
    ) % 24

    metrics = {
        "mesor": params["mesor"],
        "amplitude": params["amplitude"],
        "acrophase_hours": peak_hour,
        "r_squared": model.rsquared,
    }

    for metric, value in metrics.items():
        results.append({
            "source": "CosinorPy-3.1",
            "profile": name,
            "resolution": resolution,
            "metric": metric,
            "value": float(value),
        })

    print(f"Completed: {name} {resolution}")

reference = pd.DataFrame(results)

reference.to_csv(validation_dir / "reference_cosinor.csv", index=False)

print("\nTotal reference results:", len(reference))
print(reference.head(8).to_string(index=False))
