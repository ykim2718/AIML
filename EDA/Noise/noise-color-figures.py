"""Generate the waveform and power spectral density figures for the noise color taxonomy.

Changelog:
    0.0.0.2026.8.20 Initial release.
"""

import argparse
import enum
import pathlib
import sys
from typing import Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import signal
from tqdm import tqdm

__author__ = 'yRocket'
__version__ = "0.0.0.2026.8.20"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

matplotlib.use('Agg')


class NoiseColor(enum.StrEnum):
    """Noise color names used as figure keys."""

    WHITE = enum.auto()
    PINK = enum.auto()
    RED = enum.auto()
    BLUE = enum.auto()


# Power spectral density exponent beta in S(f) proportional to f ** beta.
# A factor of 2 in frequency changes the density by 10 * log10(2 ** beta), so beta -1 is -3 dB per octave.
PSD_EXPONENT: dict = {
    NoiseColor.WHITE: 0.0,
    NoiseColor.PINK: -1.0,
    NoiseColor.RED: -2.0,
    NoiseColor.BLUE: 1.0,
}

PLOT_COLOR: dict = {
    NoiseColor.WHITE: TABLEAU_COLORS['tab:gray'],
    NoiseColor.PINK: TABLEAU_COLORS['tab:pink'],
    NoiseColor.RED: TABLEAU_COLORS['tab:red'],
    NoiseColor.BLUE: TABLEAU_COLORS['tab:blue'],
}

PANEL_LABEL: list = ['(a)', '(b)', '(c)', '(d)']


def generate_noise(color: NoiseColor = None, n_samples: int = None, rng: np.random.Generator = None) -> np.ndarray:
    """Shape white gaussian noise in the frequency domain so that its density follows f ** beta.

    The amplitude spectrum is scaled by f ** (beta / 2) because the density is the squared amplitude. The zero
    frequency bin is left at zero so that the returned signal has no offset.
    """
    beta = PSD_EXPONENT[color]
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    frequency = np.fft.rfftfreq(n_samples, d=1.0)

    scale = np.zeros_like(frequency)
    scale[1:] = frequency[1:] ** (beta / 2.0)
    shaped = np.fft.irfft(spectrum * scale, n=n_samples)

    return shaped / np.std(shaped)


def plot_waveforms(signals: dict = None, sample_rate: float = None, out_path: pathlib.Path = None,
                   dpi: int = None) -> None:
    """Draw one panel per color, sharing the time and amplitude axes so the panels can be compared directly."""
    fig, axes = plt.subplots(nrows=len(signals), ncols=1, figsize=(9.0, 7.0), sharex=True, sharey=True)

    for axis, label, (color, wave) in zip(axes, PANEL_LABEL, signals.items()):
        time = np.arange(wave.size) / sample_rate
        axis.plot(time, wave, color=PLOT_COLOR[color], linewidth=0.9)
        axis.set_ylabel('Amplitude')
        axis.set_title(f"{label} {color.value} noise", loc='left', fontsize=10)
        axis.grid(visible=True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_psd(signals: dict = None, sample_rate: float = None, out_path: pathlib.Path = None,
             dpi: int = None) -> None:
    """Draw the Welch density of every color on log-log axes, where a power law becomes a straight line."""
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(9.0, 6.0))

    for color, wave in signals.items():
        frequency, density = signal.welch(wave, fs=sample_rate, nperseg=4096)
        keep = frequency > 0
        slope_db = 10.0 * np.log10(2.0) * PSD_EXPONENT[color]
        sign = '' if round(slope_db) == 0 else f"{slope_db:+.0f}"
        slope_text = '0 dB/octave' if round(slope_db) == 0 else f"{sign} dB/octave"
        axis.loglog(frequency[keep], density[keep], color=PLOT_COLOR[color], linewidth=1.2,
                    label=f"{color.value}  {slope_text}")

    axis.set_xlabel('Frequency (Hz)')
    axis.set_ylabel('Power spectral density')
    axis.grid(visible=True, which='both', alpha=0.3)
    axis.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_figures(out_folder: pathlib.Path = None, n_samples: int = None, n_wave_samples: int = None,
                  sample_rate: float = None, seed: int = None, dpi: int = None) -> None:
    """Generate the signals for both figures and write the figures into the output folder.

    The two figures need different signal lengths. The density needs a long signal so that Welch can average many
    segments, while the waveform needs a short one. Red noise holds most of its energy at the lowest frequency the
    signal reaches, so a short window cut out of a long signal is a nearly flat line and shows no character.
    """
    rng = np.random.default_rng(seed)

    wave_signals: dict = {}
    psd_signals: dict = {}
    pbar = tqdm(list(NoiseColor), ncols=100, unit='color')
    for color in pbar:
        pbar.set_description(f"Generating {color.value}")
        wave_signals[color] = generate_noise(color=color, n_samples=n_wave_samples, rng=rng)
        psd_signals[color] = generate_noise(color=color, n_samples=n_samples, rng=rng)

    plot_waveforms(signals=wave_signals, sample_rate=sample_rate, out_path=out_folder / 'fig1_waveform.png', dpi=dpi)
    plot_psd(signals=psd_signals, sample_rate=sample_rate, out_path=out_folder / 'fig2_psd.png', dpi=dpi)


def parse_args() -> argparse.Namespace:
    """Read the command line options and reject an output folder that cannot be used."""
    parser = argparse.ArgumentParser(description="Generate the noise color waveform and PSD figures.")
    parser.add_argument('--out-folder', type=str, required=True,
                        help="folder the figures are written into; it is created when missing")
    parser.add_argument('--n-samples', type=int, default=262144,
                        help="number of samples in the signal the density is estimated from")
    parser.add_argument('--n-wave-samples', type=int, default=1024,
                        help="number of samples in the signal the waveform figure draws")
    parser.add_argument('--sample-rate', type=float, default=8000.0,
                        help="sample rate in Hz used for the time and frequency axes")
    parser.add_argument('--seed', type=int, default=0,
                        help="seed of the random generator, so the figures are reproducible")
    parser.add_argument('--dpi', type=int, default=300,
                        help="resolution of the saved figures")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    out_folder = pathlib.Path(args.out_folder)
    if out_folder.exists() and not out_folder.is_dir():
        parser.error(f"--out-folder is not a folder: {out_folder}")
    out_folder.mkdir(parents=True, exist_ok=True)
    args.out_folder = out_folder

    if args.n_samples < 4096:
        parser.error(f"--n-samples must be at least 4096 to fill one Welch segment: {args.n_samples}")
    if args.sample_rate <= 0.0:
        parser.error(f"--sample-rate must be positive: {args.sample_rate}")
    if args.dpi <= 0:
        parser.error(f"--dpi must be positive: {args.dpi}")
    if args.n_wave_samples < 16:
        parser.error(f"--n-wave-samples must be at least 16 to show a waveform: {args.n_wave_samples}")

    return args


if __name__ == '__main__':
    cli = parse_args()
    build_figures(out_folder=cli.out_folder, n_samples=cli.n_samples, n_wave_samples=cli.n_wave_samples,
                  sample_rate=cli.sample_rate, seed=cli.seed, dpi=cli.dpi)
