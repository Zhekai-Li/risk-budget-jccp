from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> Any:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path
