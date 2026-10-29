from pathlib import Path

import pandas as pd

from forest.oak import circadian_rhythms as cr

validation_dir = Path(__file__).resolve().parent
profile_dir = validation_dir / "profiles"

resolutions = {
    "1min": (1440, 60),
    "1h": (24, 3600),
}

forest_results = []
window_notes = {}

for path in sorted(profile_dir.glob("*.csv")):
    name, resolution = path.stem.rsplit("_", 1)
    intervals_per_day, interval_seconds = resolutions[resolution]

    data = pd.read_csv(path)
    activity = data["activity"].to_numpy(dtype=float)

    windows = cr.compute_l5_m10(
        activity, intervals_per_day
    )

    fit = cr.cosinor(
        activity, interval_seconds
    )

    metrics = {
        "IS": cr.interdaily_stability(
            activity, intervals_per_day
        ),
        "IV": cr.intradaily_variability(activity),
        "L5": windows.l5,
        "M10": windows.m10,
        "RA": cr.relative_amplitude(
            windows.l5, windows.m10
        ),
        "mesor": fit.mesor,
        "amplitude": fit.amplitude,
        "acrophase_hours": fit.acrophase_hours,
        "r_squared": fit.r_squared,
    }

    window_notes[(name, resolution)] = {
        "L5": windows.l5_onset_hours + 5 > 24,
        "M10": windows.m10_onset_hours + 10 > 24,
    }

    for metric, value in metrics.items():
        forest_results.append({
            "profile": name,
            "resolution": resolution,
            "metric": metric,
            "forest_value": float(value),
        })

    print(f"Forest completed: {name} {resolution}")

forest = pd.DataFrame(forest_results)

references = pd.concat(
    [
        pd.read_csv(validation_dir / "reference_actcr.csv"),
        pd.read_csv(validation_dir / "reference_cosinor.csv"),
    ],
    ignore_index=True,
)

comparison = references.merge(
    forest,
    on=["profile", "resolution", "metric"],
    how="left",
    validate="one_to_one",
)

if len(comparison) != 108:
    raise ValueError("Expected 108 comparison rows")

if comparison["forest_value"].isna().any():
    raise ValueError("Some Forest metrics are missing")

comparison["absolute_difference"] = (
    comparison["forest_value"] - comparison["value"]
).abs()

phase = comparison["metric"] == "acrophase_hours"

comparison.loc[phase, "absolute_difference"] = (
    (
        comparison.loc[phase, "forest_value"]
        - comparison.loc[phase, "value"]
        + 12
    ) % 24 - 12
).abs()

def convention_note(row):
    if row["source"] != "ActCR-0.4.0":
        return ""

    windows = window_notes[
        (row["profile"], row["resolution"])
    ]

    if row["metric"] == "L5" and windows["L5"]:
        return "Forest L5 crosses midnight"

    if row["metric"] == "M10" and windows["M10"]:
        return "Forest M10 crosses midnight"

    if row["metric"] == "RA" and any(windows.values()):
        return "Possible circular-window difference"

    return ""

comparison["note"] = comparison.apply(
    convention_note, axis=1
)

output = validation_dir / "forest_comparison.csv"
comparison.to_csv(output, index=False)

print("\nComparison rows:", len(comparison))

print("\nMaximum absolute difference by metric:")
summary = comparison.groupby(
    ["source", "metric"]
)["absolute_difference"].max()

print(summary.to_string())

print("\nLargest 20 differences:")
columns = [
    "profile",
    "resolution",
    "metric",
    "value",
    "forest_value",
    "absolute_difference",
    "note",
]

print(
    comparison.sort_values(
        "absolute_difference",
        ascending=False
    )[columns].head(20).to_string(index=False)
)

print("\nSaved:", output)
