from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


def rounded_table(frame: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include=["number"]).columns
    result.loc[:, numeric] = result.loc[:, numeric].round(decimals)
    return result


def write_latex_table(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    decimals: int = 4,
    escape: bool = True,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = rounded_table(frame, decimals=decimals)
    try:
        content = table.to_latex(index=False, escape=escape, booktabs=True)
    except TypeError:
        content = table.to_latex(index=False, escape=escape)
    path.write_text(content, encoding="utf-8")
    return path


def copy_asset(source: str | Path, destination_dir: str | Path) -> Path:
    src = Path(source)
    dst_dir = Path(destination_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
    return dst
