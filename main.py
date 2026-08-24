# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:23:46 2026

@author: vijit
"""

from pathlib import Path
from mea_time_freq import MEATxtOpener, plot_time_frequency
import matplotlib.pyplot as plt

# Folder containing the input TXT files.
folder_path = Path("data_files")

# Store all generated files in a separate folder.
output_folder = folder_path / "time_frequency_results"
output_folder.mkdir(parents=True, exist_ok=True)

# Find all TXT files directly inside data_files.
txt_files = sorted(folder_path.glob("*.txt"))

if not txt_files:
    raise FileNotFoundError(
        f"No TXT files were found in: {folder_path.resolve()}"
    )

print(f"Found {len(txt_files)} TXT file(s).")


for file_number, input_path in enumerate(txt_files, start=1):
    print(
        f"\nProcessing {file_number}/{len(txt_files)}: "
        f"{input_path.name}"
    )

    png_output_path = output_folder / (
        f"{input_path.stem}_time_frequency.png"
    )

    csv_output_path = output_folder / (
        f"{input_path.stem}_time_frequency.csv"
    )

    try:
        recording = MEATxtOpener(input_path).load()

        print(f"Detected columns: {recording.columns}")
        
        ## Change by Aisha ##
        figure, axes = plot_time_frequency(
            recording,
            time_column=0,
            signal_column=2,
            start_seconds=None,
            end_seconds=None,
            min_frequency=1,
            max_frequency=250,
            window_seconds=0.1,
            overlap_fraction=0.90,
            relative_db=True,
            csv_output_path=csv_output_path,
        )
        ##              ##

        figure.savefig(
            png_output_path,
            dpi=300,
            bbox_inches="tight",
        )

        # Release the figure's memory before processing the next file.
        plt.close(figure)

        print(f"Saved PNG: {png_output_path.resolve()}")
        print(f"Saved CSV: {csv_output_path.resolve()}")

    except Exception as error:
        # Report the failure but continue processing the remaining files.
        print(f"Failed to process {input_path.name}")
        print(f"Reason: {error}")


print("\nFinished processing all TXT files.")