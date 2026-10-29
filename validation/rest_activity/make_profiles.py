from pathlib import Path

import numpy as np
import pandas as pd

output = Path(__file__).resolve().parent / "profiles"
output.mkdir(exist_ok=True)

rng = np.random.default_rng(20261028)
minutes_per_day = 1440
n_days = 7
n = n_days * minutes_per_day

time_index = np.arange(n)
hours = (time_index % minutes_per_day) / 60

def cosine(peak):
    return 0.05 + 0.04 * np.cos(
        2 * np.pi * (hours - peak) / 24
    )

square = np.where(
    (hours >= 8) & (hours < 22), 0.08, 0.02
)

naps = square.copy()

for day in range(n_days):
    for _ in range(2):
        start = int(rng.integers(8 * 60, 21 * 60))
        duration = int(rng.integers(15, 46))
        end = min(start + duration, 22 * 60)

        naps[
            day * minutes_per_day + start:
            day * minutes_per_day + end
        ] = 0.02

profiles = {
    "cosine_16": cosine(16),
    "cosine_14": cosine(14),
    "noisy_cosine": np.maximum(
        0, cosine(16) + rng.normal(0, 0.005, n)
    ),
    "square_wave": square,
    "white_noise": np.maximum(
        0, rng.normal(0.05, 0.02, n)
    ),
    "square_with_naps": naps,
}

timestamps = pd.date_range(
    "2026-02-01", periods=n, freq="min"
)

for name, activity in profiles.items():
    minute_data = pd.DataFrame({
        "timestamp": timestamps,
        "activity": activity,
    })

    hourly_data = pd.DataFrame({
        "timestamp": timestamps[::60],
        "activity": activity.reshape(-1, 60).mean(axis=1),
    })

    minute_data.to_csv(
        output / f"{name}_1min.csv", index=False
    )
    hourly_data.to_csv(
        output / f"{name}_1h.csv", index=False
    )

    print(f"{name}: 10080 minute samples, 168 hourly samples")
