"""Generate the waveform, power spectral density and point pattern figures for the noise color taxonomy.

Changelog:
    0.2.0.2026.8.22 Fit the density of every color on log-log axes and report the fitted slope in dB per octave.
    0.1.0.2026.8.21 Add the point pattern figure that contrasts white and blue noise sampling.
    0.0.0.2026.8.20 Initial release.
"""

import argparse
import csv
import enum
import pathlib
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import signal
from scipy.spatial import distance
from tqdm import tqdm

__author__ = 'yRocket'
__version__ = "0.2.0.2026.8.22"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

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

# Decibels the density moves when the frequency doubles, for an exponent of one. It is 10 * log10(2).
DB_PER_OCTAVE: float = 10.0 * np.log10(2.0)

# Header of the file that keeps the fitted slopes, so the numbers quoted elsewhere can be recomputed.
FIT_FIELD: list = ['color', 'nominal_exponent', 'fitted_exponent', 'nominal_slope_db_per_octave',
                   'fitted_slope_db_per_octave', 'slope_stderr_db_per_octave', 'fit_low_hz', 'fit_high_hz',
                   'n_fit_points']


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


def format_slope(slope_db: float = None) -> str:
    """Write a slope in dB per octave, keeping the sign only when the slope is not flat."""
    return '0 dB/octave' if round(slope_db) == 0 else f"{slope_db:+.0f} dB/octave"


def fit_psd_slope(frequency: np.ndarray = None, density: np.ndarray = None, fit_low: float = None,
                  fit_high: float = None) -> dict:
    """Fit a straight line to the density on log-log axes and return the slope with its standard error.

    A power law is a straight line once both axes are logarithmic, so the exponent is the slope of that line and
    the slope in dB per octave is the exponent times 10 * log10(2). The fit runs over the band between fit_low and
    fit_high because the lowest bins average too few segments and the highest bins sit on the filter edge.

    Returns a dict with the fitted exponent, the slope and its standard error in dB per octave, the frequency of
    the fitted points and the fitted density, so the caller can both quote and draw the fit.
    """
    band = (frequency >= fit_low) & (frequency <= fit_high)
    if band.sum() < 3:
        raise ValueError(f"fit band holds {band.sum()} points, which is too few to fit a line and its error: "
                         f"{fit_low=}, {fit_high=}")

    log_frequency = np.log10(frequency[band])
    log_density = np.log10(density[band])
    coefficient, covariance = np.polyfit(log_frequency, log_density, deg=1, cov=True)

    return {
        'exponent': float(coefficient[0]),
        'slope_db': float(coefficient[0]) * DB_PER_OCTAVE,
        'stderr_db': float(np.sqrt(covariance[0, 0])) * DB_PER_OCTAVE,
        'frequency': frequency[band],
        'density': 10.0 ** np.polyval(coefficient, log_frequency),
        'n_points': int(band.sum()),
    }


def write_slope_fits(fits: dict = None, fit_low: float = None, fit_high: float = None,
                     out_path: pathlib.Path = None) -> None:
    """Write one row per color so that every slope drawn in the figure can be read back as a number."""
    with out_path.open(mode='w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIT_FIELD)
        writer.writeheader()
        for color, fit in fits.items():
            writer.writerow({
                'color': color.value,
                'nominal_exponent': f"{PSD_EXPONENT[color]:.1f}",
                'fitted_exponent': f"{fit['exponent']:.4f}",
                'nominal_slope_db_per_octave': f"{PSD_EXPONENT[color] * DB_PER_OCTAVE:.4f}",
                'fitted_slope_db_per_octave': f"{fit['slope_db']:.4f}",
                'slope_stderr_db_per_octave': f"{fit['stderr_db']:.4f}",
                'fit_low_hz': f"{fit_low:.4f}",
                'fit_high_hz': f"{fit_high:.4f}",
                'n_fit_points': fit['n_points'],
            })


def plot_waveforms(signals: dict = None, sample_rate: float = None, out_path: pathlib.Path = None,
                   dpi: int = None) -> None:
    """Draw one panel per color, sharing the time and amplitude axes so the panels can be compared directly."""
    fig, axes = plt.subplots(nrows=len(signals), ncols=1, figsize=(9.0, 7.0), sharex=True, sharey=True)

    for axis, label, (color, wave) in zip(axes, PANEL_LABEL, signals.items()):
        time = np.arange(wave.size) / sample_rate
        axis.plot(time, wave, color=PLOT_COLOR[color], linewidth=0.9)
        axis.set_ylabel('Amplitude')
        slope_text = format_slope(slope_db=PSD_EXPONENT[color] * DB_PER_OCTAVE)
        axis.set_title(f"{label} {color.value} noise  |  {slope_text}", loc='left', fontsize=10)
        axis.grid(visible=True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_psd(signals: dict = None, sample_rate: float = None, n_per_segment: int = None, fit_low: float = None,
             fit_high: float = None, out_path: pathlib.Path = None, dpi: int = None) -> dict:
    """Draw the Welch density of every color on log-log axes, where a power law becomes a straight line.

    Every density is fitted over the band between fit_low and fit_high and the fitted line is drawn on top of it,
    so the slope the eye reads off the curve can be compared with the slope the fit returns. The legend carries
    the nominal slope of the color and the fitted slope with its standard error. Returns the fit of every color.
    """
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(9.0, 6.0))

    fits: dict = {}
    last_index = len(signals) - 1
    for index, (color, wave) in enumerate(signals.items()):
        frequency, density = signal.welch(wave, fs=sample_rate, nperseg=n_per_segment)
        keep = frequency > 0
        fits[color] = fit_psd_slope(frequency=frequency[keep], density=density[keep], fit_low=fit_low,
                                    fit_high=fit_high)

        nominal_text = format_slope(slope_db=PSD_EXPONENT[color] * DB_PER_OCTAVE)
        fitted_text = f"fit {fits[color]['slope_db']:+.2f} ± {fits[color]['stderr_db']:.2f} dB/octave"
        axis.loglog(frequency[keep], density[keep], color=PLOT_COLOR[color], linewidth=1.2,
                    label=f"{color.value}  {nominal_text}  ({fitted_text})")
        axis.loglog(fits[color]['frequency'], fits[color]['density'], color='black', linestyle='--',
                    linewidth=1.3, label='least squares fit' if index == last_index else '_nolegend_')

    axis.axvspan(fit_low, fit_high, color='black', alpha=0.05, label=f"fit band {fit_low:g} to {fit_high:g} Hz")
    axis.set_xlabel('Frequency (Hz)')
    axis.set_ylabel('Power spectral density')
    axis.grid(visible=True, which='both', alpha=0.3)
    axis.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return fits


def generate_uniform_points(n_points: int = None, rng: np.random.Generator = None) -> np.ndarray:
    """Place points independently and uniformly, which is the two dimensional form of white noise."""
    return rng.random(size=(n_points, 2))


def generate_blue_noise_points(n_points: int = None, n_candidates: int = None,
                               rng: np.random.Generator = None) -> np.ndarray:
    """Place points with Mitchell best candidate sampling, which suppresses the low frequency clumping.

    Every new point is chosen from n_candidates uniform draws, keeping the one whose nearest accepted point is
    farthest away. Rejecting the candidates that land close to an accepted point is what removes the clumps.
    """
    points = np.empty(shape=(n_points, 2))
    points[0] = rng.random(size=2)

    for index in range(1, n_points):
        candidates = rng.random(size=(n_candidates, 2))
        nearest = distance.cdist(candidates, points[:index]).min(axis=1)
        points[index] = candidates[int(np.argmax(nearest))]

    return points


def plot_point_patterns(patterns: dict = None, out_path: pathlib.Path = None, dpi: int = None) -> None:
    """Draw one square panel per pattern so that the clumping can be compared at the same point count."""
    fig, axes = plt.subplots(nrows=1, ncols=len(patterns), figsize=(9.0, 4.8))

    for axis, label, (color, points) in zip(axes, PANEL_LABEL, patterns.items()):
        axis.scatter(points[:, 0], points[:, 1], s=6.0, color=PLOT_COLOR[color])
        axis.set_title(f"{label} {color.value} noise", loc='left', fontsize=10)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect('equal')
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_figures(out_folder: pathlib.Path = None, n_samples: int = None, n_wave_samples: int = None,
                  sample_rate: float = None, n_per_segment: int = None, fit_low: float = None,
                  fit_high: float = None, n_points: int = None, n_candidates: int = None,
                  seed: int = None, dpi: int = None) -> None:
    """Generate the signals for both figures and write the figures into the output folder.

    The two figures need different signal lengths. The density needs a long signal so that Welch can average many
    segments, while the waveform needs a short one. Red noise holds most of its energy at the lowest frequency the
    signal reaches, so a short window cut out of a long signal is a nearly flat line and shows no character.

    The fitted slopes are written next to the figures so that the numbers drawn in the density figure can be read
    back as values instead of being copied off the picture.
    """
    rng = np.random.default_rng(seed)

    wave_signals: dict = {}
    psd_signals: dict = {}
    pbar = tqdm(list(NoiseColor), ncols=100, unit='color')
    for color in pbar:
        pbar.set_description(f"Generating {color.value}")
        wave_signals[color] = generate_noise(color=color, n_samples=n_wave_samples, rng=rng)
        psd_signals[color] = generate_noise(color=color, n_samples=n_samples, rng=rng)

    patterns: dict = {
        NoiseColor.WHITE: generate_uniform_points(n_points=n_points, rng=rng),
        NoiseColor.BLUE: generate_blue_noise_points(n_points=n_points, n_candidates=n_candidates, rng=rng),
    }

    plot_waveforms(signals=wave_signals, sample_rate=sample_rate, out_path=out_folder / 'fig1_waveform.png', dpi=dpi)
    fits = plot_psd(signals=psd_signals, sample_rate=sample_rate, n_per_segment=n_per_segment, fit_low=fit_low,
                    fit_high=fit_high, out_path=out_folder / 'fig2_psd.png', dpi=dpi)
    plot_point_patterns(patterns=patterns, out_path=out_folder / 'fig3_point_pattern.png', dpi=dpi)
    write_slope_fits(fits=fits, fit_low=fit_low, fit_high=fit_high, out_path=out_folder / 'fig2_psd_slope_fit.csv')


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
    parser.add_argument('--n-per-segment', type=int, default=4096,
                        help="samples per Welch segment; longer resolves low frequencies, shorter averages more")
    parser.add_argument('--fit-low-hz', type=float, default=10.0,
                        help="lower edge of the band the density is fitted over; below it Welch averages too few "
                             "segments")
    parser.add_argument('--fit-high-hz', type=float, default=2000.0,
                        help="upper edge of the band the density is fitted over; keep it below the Nyquist edge")
    parser.add_argument('--n-points', type=int, default=800,
                        help="number of points drawn in each panel of the point pattern figure")
    parser.add_argument('--n-candidates', type=int, default=12,
                        help="candidates drawn per accepted point in blue noise sampling; larger is more even")
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

    if args.sample_rate <= 0.0:
        parser.error(f"--sample-rate must be positive: {args.sample_rate}")
    if args.n_per_segment < 256:
        parser.error(f"--n-per-segment must be at least 256 to resolve a slope: {args.n_per_segment}")
    if args.n_samples < args.n_per_segment:
        parser.error(f"--n-samples must fill at least one Welch segment: {args.n_samples} < {args.n_per_segment}")
    if args.fit_low_hz <= 0.0:
        parser.error(f"--fit-low-hz must be positive, since a log axis has no zero: {args.fit_low_hz}")
    if args.fit_high_hz <= args.fit_low_hz:
        parser.error(f"--fit-high-hz must be above --fit-low-hz: {args.fit_high_hz} <= {args.fit_low_hz}")
    if args.fit_high_hz > args.sample_rate / 2.0:
        parser.error(f"--fit-high-hz must stay at or below the Nyquist frequency {args.sample_rate / 2.0}: "
                     f"{args.fit_high_hz}")
    if args.dpi <= 0:
        parser.error(f"--dpi must be positive: {args.dpi}")
    if args.n_wave_samples < 16:
        parser.error(f"--n-wave-samples must be at least 16 to show a waveform: {args.n_wave_samples}")
    if args.n_points < 2:
        parser.error(f"--n-points must be at least 2: {args.n_points}")
    if args.n_candidates < 2:
        parser.error(f"--n-candidates must be at least 2, otherwise the sampling is uniform: {args.n_candidates}")

    return args


if __name__ == '__main__':
    cli = parse_args()
    build_figures(out_folder=cli.out_folder, n_samples=cli.n_samples, n_wave_samples=cli.n_wave_samples,
                  sample_rate=cli.sample_rate, n_per_segment=cli.n_per_segment, fit_low=cli.fit_low_hz,
                  fit_high=cli.fit_high_hz, n_points=cli.n_points, n_candidates=cli.n_candidates,
                  seed=cli.seed, dpi=cli.dpi)
