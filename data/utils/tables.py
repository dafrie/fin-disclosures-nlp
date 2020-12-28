import os
import pandas as pd

import config as cfg

DEFAULT_EXPORT_PATH = os.path.join(cfg.LATEX_EXPORT_PATH, 'tables/')


def add_hline(latex: str, index: int) -> str:
    """
    Adds a horizontal `index` lines before the last line of the table

    Args:
        latex: latex table
        index: index of horizontal line insertion (in lines)
    """
    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines).replace('NaN', '')


def add_bold_line(latex: str, index: int) -> str:
    """Makes a provided line number bold
    """
    lines = latex.splitlines()
    cells = lines[index].split("&")
    lines[index] = r'\bfseries ' + r'& \bfseries '.join(cells)
    return '\n'.join(lines)


def export_to_latex(df=None, latex_str=None, filename=None, path=None, add_midrule_at=0, add_verticalrule_at=None, make_bold_row_at=None, correct_multicolumn=False, **kwargs):
    if not latex_str:
        latex_str = df.to_latex(**kwargs)

    if add_midrule_at:
        latex_str = add_hline(latex_str, add_midrule_at)

    if make_bold_row_at is not None:
        rows = make_bold_row_at if isinstance(make_bold_row_at, list) else [
            make_bold_row_at]
        for r in rows:
            latex_str = add_bold_line(latex_str, r)

    if add_verticalrule_at is not None:
        len_start = len("\begin{tabular}")
        start = latex_str.find("{", len_start)
        end = latex_str.find("}", len_start+1)

        col_defs = latex_str[start+1:end]
        for c in reversed(add_verticalrule_at):
            col_defs = col_defs[:c] + "|" + col_defs[c:]
        latex_str = latex_str[:start+1] + col_defs + latex_str[end:]

    if correct_multicolumn:
        latex_str = latex_str.replace("\\toprule", "\\toprule &")

    if not path:
        path = os.path.join(DEFAULT_EXPORT_PATH, filename)

    with open(path, 'w') as tf:
        tf.write(latex_str)
    return latex_str
