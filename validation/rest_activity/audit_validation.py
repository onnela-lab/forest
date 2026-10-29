from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / "forest_comparison.csv")

tolerances = {
    "IS": 1e-10,
    "IV": 1e-10,
    "L5": 1e-6,
    "M10": 1e-6,
    "RA": 1e-6,
    "mesor": 1e-6,
    "amplitude": 1e-6,
    "acrophase_hours": 0.025,
    "r_squared": 1e-10,
}


def classify(row):
    if not np.isfinite(row["value"]) or not np.isfinite(
        row["forest_value"]
    ):
        return "REVIEW"

    if (
        row["profile"] == "white_noise"
        and row["metric"] == "acrophase_hours"
    ):
        return "LOW_SIGNAL_PHASE"

    if row["absolute_difference"] <= tolerances[row["metric"]]:
        return "AGREE"

    if (
        row["source"] == "ActCR-0.4.0"
        and row["metric"] in {"L5", "M10", "RA"}
        and isinstance(row["note"], str)
        and row["note"]
    ):
        return "DEFINITION_DIFFERENCE"

    return "REVIEW"


data["status"] = data.apply(classify, axis=1)

print("\nValidation classifications:")
print(data["status"].value_counts().to_string())

print("\nRows requiring investigation:")
review = data[data["status"] == "REVIEW"]
print(
    "None" if review.empty else review.to_string(index=False)
)

print("\nDocumented convention differences:")
differences = data[
    data["status"] == "DEFINITION_DIFFERENCE"
]

print(
    differences[
        [
            "profile",
            "resolution",
            "metric",
            "absolute_difference",
            "note",
        ]
    ].to_string(index=False)
)

output = folder / "validation_audit.csv"
data.to_csv(output, index=False)

print("\nSaved:", output)

if not review.empty:
    raise SystemExit(1)
