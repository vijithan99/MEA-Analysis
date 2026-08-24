# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:23:46 2026

@author: vijit
"""

from pathlib import Path
from mea_time_freq import MEATxtOpener, plot_time_frequency
import matplotlib.pyplot as plt

input_path = Path(
    "50-4AP testing channel 17.txt"
)

png_output_path = input_path.with_name(
    f"{input_path.stem}_time_frequency.png"
)

csv_output_path = input_path.with_name(
    f"{input_path.stem}_time_frequency.csv"
)

recording = MEATxtOpener(input_path).load()
print(recording.columns)

figure, axes = plot_time_frequency(
    recording,
    time_column=0,
    signal_column=1,
    start_seconds=None,
    end_seconds=None,
    min_frequency=1,
    max_frequency=250,
    window_seconds=0.5,
    overlap_fraction=0.90,
    relative_db=True,
    csv_output_path=csv_output_path,
)

figure.savefig(
    png_output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved figure: {png_output_path.resolve()}")

plt.show()