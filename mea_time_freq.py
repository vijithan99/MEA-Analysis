# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:23:46 2026

@author: vijit
"""

import argparse
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

@dataclass
class MEATxtOpener():
    file_path: str | Path
    skip_rows: int = 1
    comment_prefixes: tuple[str, ...] = ("#", "%")
    data: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def load(self) -> "MEATxtOpener":
        path = Path(self.file_path)
    
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
    
        # Read only the header.
        with path.open("r", encoding="utf-8-sig") as file:
            header_line = file.readline().strip()
    
        headers = [
            header.strip().strip('"')
            for header in header_line.split("\t")
        ]
    
        # Load the numeric rows efficiently.
        matrix = np.loadtxt(
            path,
            delimiter="\t",
            skiprows=1,
        )
    
        if matrix.shape[1] != len(headers):
            raise ValueError(
                f"Found {len(headers)} headers but "
                f"{matrix.shape[1]} numeric columns."
            )
    
        self.data = {
            header: matrix[:, column_index]
            for column_index, header in enumerate(headers)
        }
    
        return self

    @property
    def columns(self) -> list[str]:
        return list(self.data)

    def column(self, selector: int | str) -> tuple[str, np.ndarray]:
        """Select a column by zero-based index, exact name, or unique partial name."""
        if not self.data:
            raise RuntimeError("Call load() before selecting a column.")

        if isinstance(selector, int):
            try:
                name = self.columns[selector]
            except IndexError as error:
                raise IndexError(
                    f"Column index {selector} is invalid. Available columns: "
                    f"{list(enumerate(self.columns))}"
                ) from error
            return name, self.data[name]

        normalized = selector.casefold()
        exact = [name for name in self.columns if name.casefold() == normalized]
        if exact:
            return exact[0], self.data[exact[0]]

        partial = [name for name in self.columns if normalized in name.casefold()]
        if len(partial) == 1:
            return partial[0], self.data[partial[0]]
        if len(partial) > 1:
            raise KeyError(f"'{selector}' matches multiple columns: {partial}")
        raise KeyError(f"Column '{selector}' was not found. Available: {self.columns}")

    def sampling_rate_calc(self, time_column: int | str = 0) -> float:
        _, time = self.column(time_column)
        sampling_interval = np.median(np.diff(time))
        sampling_rate = 1 / sampling_interval
        
        return sampling_rate

def _selector(value: str) -> int | str:
    """Treat a whole-number command-line selector as a column index."""
    return int(value) if re.fullmatch(r"\d+", value) else value

def _pathchange(path: str | Path, file_type: str):
    path = Path(path)
    file_type = file_type.lstrip(".")
    
    return path.with_name(f"{path.stem}_time_frequency.{file_type}")

def save_time_frequency_csv(
    output_path: str | Path,
    time: np.ndarray,
    frequencies: np.ndarray,
    power_db: np.ndarray,
    power_column_name: str,
) -> None:

    output_path = Path(output_path)

    number_of_times = len(time)
    number_of_frequencies = len(frequencies)

    # Repeat each time for every frequency.
    csv_time = np.repeat(
        time,
        number_of_frequencies,
    )

    # Repeat the frequency array for every time window.
    csv_frequency = np.tile(
        frequencies,
        number_of_times,
    )

    # Transpose so the ordering matches time, then frequency.
    csv_power = power_db.T.ravel()

    csv_data = np.column_stack(
        (
            csv_time,
            csv_frequency,
            csv_power,
        )
    )

    np.savetxt(
        output_path,
        csv_data,
        delimiter=",",
        header=f"Time_s,Frequency_Hz,{power_column_name}",
        comments="",
        fmt=("%.6f", "%.3f", "%.6f"),
    )

    print(f"Saved time-frequency CSV: {output_path.resolve()}")

def plot_time_frequency(
    recording: MEATxtOpener,
    time_column: int | str = 0,
    signal_column: int | str = 1,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    min_frequency: float = 1.0,
    max_frequency: float | None = 250.0,
    window_seconds: float = 0.5,
    overlap_fraction: float = 0.90,
    relative_db: bool = False,
    csv_output_path: str | Path | None = None,    
    cmap: str = "magma",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the selected MEA trace and its short-time Fourier spectrogram.

    ``window_seconds`` controls the time/frequency trade-off. A longer window
    improves frequency resolution but blurs short events in time.
    """
    time_name, time = recording.column(time_column)
    signal_name, voltage = recording.column(signal_column)

    valid = np.isfinite(time) & np.isfinite(voltage)
    if start_seconds is not None:
        valid &= time >= start_seconds
    if end_seconds is not None:
        valid &= time <= end_seconds

    time = time[valid]
    voltage = voltage[valid]
    if time.size < 8:
        raise ValueError("The selected interval contains fewer than eight samples.")

    sample_rate = recording.sampling_rate_calc(time_column)
    nyquist = sample_rate / 2.0
    upper_frequency = nyquist if max_frequency is None else min(max_frequency, nyquist)
    if not 0 <= min_frequency < upper_frequency:
        raise ValueError(
            f"Frequency limits must satisfy 0 <= min < max <= Nyquist "
            f"({nyquist:g} Hz)."
        )
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be at least 0 and less than 1.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    # Remove the DC level before computing the short-time spectrum.
    centered_voltage = scipy_signal.detrend(voltage, type="constant")
    samples_per_window = min(
        max(8, int(round(window_seconds * sample_rate))),
        centered_voltage.size,
    )
    overlap_samples = min(
        int(round(overlap_fraction * samples_per_window)),
        samples_per_window - 1,
    )

    frequencies, segment_times, power = scipy_signal.spectrogram(
        centered_voltage,
        fs=sample_rate,
        window="hann",
        nperseg=samples_per_window,
        noverlap=overlap_samples,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    colorbar_label = "Power spectral density (dB/Hz)"

    if relative_db:
        # Highlights transient changes by removing each frequency's median power.
        power_db = power_db - np.median(power_db, axis=1, keepdims=True)
        colorbar_label = "Power relative to temporal median (dB)"

    frequency_mask = (
        (frequencies >= min_frequency) & (frequencies <= upper_frequency)
    )
    plot_time = segment_times + time[0]
    
    selected_frequencies = frequencies[frequency_mask]
    selected_power_db = power_db[frequency_mask, :]
    
    if relative_db:
        power_column_name = "Relative_power_dB"
    else:
        power_column_name = "PSD_dB_per_Hz"
    
    if csv_output_path is not None:
        save_time_frequency_csv(
            output_path=csv_output_path,
            time=plot_time,
            frequencies=selected_frequencies,
            power_db=selected_power_db,
            power_column_name=power_column_name,
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
        constrained_layout=True,
    )
    axes[0].plot(time, voltage, color="black", linewidth=0.65)
    axes[0].set_ylabel(signal_name)
    axes[0].set_title(
        f"{signal_name} | sampling rate = {sample_rate:,.2f} Hz | "
        f"window = {samples_per_window / sample_rate:.3f} s"
    )
    axes[0].grid(alpha=0.2)

    image = axes[1].pcolormesh(
        plot_time,
        selected_frequencies,
        selected_power_db,
        shading="auto",
        cmap=cmap,
    )
    
    axes[1].set_xlabel(time_name)
    axes[1].set_ylabel("Frequency (Hz)")
    figure.colorbar(image, ax=axes[1], label=colorbar_label)

    return figure, axes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a time-frequency plot from an MEA text export."
    )
    parser.add_argument("file", type=Path, help="Path to the MEA text file")
    parser.add_argument("--time-column", default="0", help="Time column name or index")
    parser.add_argument(
        "--signal-column", default="1", help="Signal column name or index"
    )
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--start", type=float, default=None, help="Start time")
    parser.add_argument("--end", type=float, default=None, help="End time")
    parser.add_argument("--min-freq", type=float, default=1.0)
    parser.add_argument("--max-freq", type=float, default=250.0)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=0.90)
    parser.add_argument(
        "--relative-db",
        action="store_true",
        help="Show change from each frequency's temporal median",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()

    recording = MEATxtOpener(arguments.file, skip_rows=arguments.skip_rows).load()
    print("Detected columns:")
    for index, name in enumerate(recording.columns):
        print(f"  {index}: {name}")
    
    csv_output_path = _pathchange(
        arguments.file,
        "csv",
    )
        
    figure, _ = plot_time_frequency(
        recording,
        time_column=_selector(arguments.time_column),
        signal_column=_selector(arguments.signal_column),
        start_seconds=arguments.start,
        end_seconds=arguments.end,
        min_frequency=arguments.min_freq,
        max_frequency=arguments.max_freq,
        window_seconds=arguments.window_seconds,
        overlap_fraction=arguments.overlap,
        relative_db=arguments.relative_db,
        csv_output_path=csv_output_path,
    )

    output_path = arguments.output
    if output_path is None:
        output_path = arguments.file.with_name(
            f"{arguments.file.stem}_time_frequency.png"
        )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path.resolve()}")

    if arguments.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
