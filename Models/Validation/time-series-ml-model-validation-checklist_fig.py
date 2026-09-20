"""Draw where each check of the validation checklist sits in the modeling pipeline.

The figure lays the six stages of the pipeline from left to right and prints the checks of each
stage under its box. The marker colour separates the time-series level from the general level.
"""
__author__ = 'yRocket'
__version__ = "0.0.1+20260920"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt

__all__ = ['pipeline_stages', 'draw_pipeline_stages']

FIGSIZE: tuple = (13.5, 3.2)
REFERENCE_WIDTH: float = 13.5        # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
COLORS: dict = {'Time-series': list(matplotlib.colors.TABLEAU_COLORS.values())[0],
                'General': list(matplotlib.colors.TABLEAU_COLORS.values())[1]}
STAGE_ORDER: tuple = ('Data split', 'Preprocessing', 'Feature engineering', 'Training',
                      'Evaluation', 'Deployment')


def pipeline_stages() -> dict:
    """The checks of Table 1, grouped by stage in the order the pipeline runs.

    Returns a dict keyed by stage name, each value a list of (check, level) pairs.

    >>> stages = pipeline_stages()
    >>> list(stages)[0], stages['Training']
    ('Data split', [('Overfitting / underfitting', 'General')])
    >>> sum(len(checks) for checks in stages.values())
    12
    """
    rows = [('Time-Series Split', 'Time-series', 'Data split'),
            ('Temporal separation of sets', 'Time-series', 'Data split'),
            ('Scaling and imputation', 'Time-series', 'Preprocessing'),
            ('Stationarity and differencing', 'Time-series', 'Preprocessing'),
            ('Lag and rolling window', 'Time-series', 'Feature engineering'),
            ('Feature leakage', 'General', 'Feature engineering'),
            ('Overfitting / underfitting', 'General', 'Training'),
            ('Metric selection', 'General', 'Evaluation'),
            ('Baseline comparison', 'General', 'Evaluation'),
            ('Concept drift', 'Time-series', 'Evaluation'),
            ('Look-ahead bias', 'Time-series', 'Deployment'),
            ('Reproducibility and latency', 'General', 'Deployment')]

    stages = {stage: [] for stage in STAGE_ORDER}
    for check, level, stage in rows:
        if stage not in stages:
            raise ValueError(f"stage '{stage}' of check '{check}' is not in STAGE_ORDER.")
        if level not in COLORS:
            raise ValueError(f"level '{level}' of check '{check}' has no colour.")
        stages[stage].append((check, level))
    return stages


def draw_pipeline_stages(stages: dict, output_folder: pathlib.Path) -> pathlib.Path:
    """Draw the stages left to right and write the figure as a png.

    stages: mapping of stage name to its (check, level) pairs, as pipeline_stages returns
    output_folder: folder the png is written to, created when it is missing

    Returns the path of the written file.
    """
    if not stages:
        raise ValueError("stages is empty; nothing to draw.")
    output_folder.mkdir(parents=True, exist_ok=True)

    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.set_xlim(0, len(stages))
    axes.set_ylim(0.34, 1.02)
    axes.axis('off')

    box_top, box_bottom = 0.86, 0.72
    for index, (stage, checks) in enumerate(stages.items()):
        left, right = index + 0.06, index + 0.94
        axes.add_patch(plt.Rectangle((left, box_bottom), right - left, box_top - box_bottom,
                                     facecolor='#eef2f5', edgecolor='#4c5966', linewidth=1.0))
        axes.text((left + right) / 2, (box_top + box_bottom) / 2, stage, ha='center', va='center',
                  fontsize=font_size * 1.05, fontweight='bold', color='#1b2430')
        axes.text((left + right) / 2, box_top + 0.05, f"Stage {index + 1}", ha='center', va='center',
                  fontsize=font_size * 0.85, color='#5d6874')

        if index < len(stages) - 1:
            axes.annotate('', xy=(index + 1.06, (box_top + box_bottom) / 2),
                          xytext=(right, (box_top + box_bottom) / 2),
                          arrowprops=dict(arrowstyle='-|>', color='#4c5966', linewidth=1.2))

        for check_index, (check, level) in enumerate(checks):
            height = box_bottom - 0.10 - check_index * 0.09
            axes.plot([left + 0.02], [height], marker='o', markersize=font_size * 0.55,
                      color=COLORS[level])
            axes.text(left + 0.08, height, check, ha='left', va='center', fontsize=font_size * 0.9,
                      color='#1b2430')

    handles = [plt.Line2D([], [], marker='o', linestyle='none', color=colour, label=f"{level} check")
               for level, colour in COLORS.items()]
    axes.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
                fontsize=font_size * 0.9, bbox_to_anchor=(0.5, -0.06))

    figure.tight_layout()
    output_path = output_folder / 'pipeline-stages.png'
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    """Read the output folder from the command line."""
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    f"Draw the pipeline stages of the validation checklist.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help="folder the figure is written to")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    written = draw_pipeline_stages(stages=pipeline_stages(), output_folder=arguments.output_folder)
    print(f"wrote {written}")
