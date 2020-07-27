import os
import re
from pathlib import Path

import matplotlib

import config as cfg

DEFAULT_EXPORT_PATH = os.path.join(cfg.LATEX_EXPORT_PATH, 'figures')
DEFAULT_FILE_FORMAT = '.pgf'


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    Credits: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def export_to_latex(plot, filename, path=DEFAULT_EXPORT_PATH, **kwargs):
    """Exports a matplotlib plot or figure to the configured export directory

    Parameters
    ----------
    plot : matplotlib.figure.Figure or matplotlib.plot
        The instance to export
    filename : string
        The filename for the exported file. Optionally include the extension (by default uses PDF)
    path : string, optional
        Override the default configured default export path, by default DEFAULT_EXPORT_PATH
    """

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    if isinstance(plot, matplotlib.figure.Figure):
        fig = plot
    else:
        fig = plot.get_figure()

    # Safely create dir if not exists already
    Path(path).mkdir(parents=True, exist_ok=True)

    # Create path, appending file suffix if not exists already
    has_extension = re.match(".*\.\w+", filename)
    if has_extension:
        path = os.path.join(path, filename)
    else:
        path = os.path.join(path, filename + DEFAULT_FILE_FORMAT)
    fig.savefig(path, **kwargs)
