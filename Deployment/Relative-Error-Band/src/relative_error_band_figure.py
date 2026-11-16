"""Save a matplotlib figure as SVG by default, and as PNG only when the SVG would be larger than the PNG.

The figure is rendered to both formats in memory, the smaller file is written to <stem>.svg or <stem>.png, and a file
of the other format left by an earlier run is removed so that the folder holds one version of the figure.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261116"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import io
import pathlib

from matplotlib import pyplot as plt

__all__ = ['save_svg_or_png']

SVG_SUFFIX: str = '.svg'
PNG_SUFFIX: str = '.png'


def save_svg_or_png(fig: plt.Figure = None, stem: pathlib.Path = None, dpi: int = None) -> pathlib.Path:
    """Write fig to stem + '.svg', or to stem + '.png' when the SVG is larger; return the written path.

    dpi applies to the PNG only. stem must not carry a suffix.
    """
    if fig is None or stem is None or dpi is None:
        raise ValueError(f"fig, stem and dpi are required: {fig=}, {stem=}, {dpi=}")
    if stem.suffix in (SVG_SUFFIX, PNG_SUFFIX):
        raise ValueError(f"stem must not carry a figure suffix: {stem}")
    rendered = {}
    for suffix, fmt in ((SVG_SUFFIX, 'svg'), (PNG_SUFFIX, 'png')):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi)
        rendered[suffix] = buffer.getvalue()
    chosen = SVG_SUFFIX if len(rendered[SVG_SUFFIX]) <= len(rendered[PNG_SUFFIX]) else PNG_SUFFIX
    other = PNG_SUFFIX if chosen == SVG_SUFFIX else SVG_SUFFIX
    path = stem.with_name(stem.name + chosen)
    path.write_bytes(rendered[chosen])
    stale = stem.with_name(stem.name + other)
    if stale.exists():
        stale.unlink()
        print(f"removed {stale}, replaced by {path.name}")
    print(f"wrote {path} (SVG {len(rendered[SVG_SUFFIX]):,} bytes, PNG {len(rendered[PNG_SUFFIX]):,} bytes)")
    return path
