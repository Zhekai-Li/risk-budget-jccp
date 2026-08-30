from __future__ import annotations

import json
from pathlib import Path
import zipfile
from urllib.request import urlopen

from risk_budget_jccp.real_data.common.downloading import download_url, run_command, write_provenance
from risk_budget_jccp.real_data.common.paths import case_raw_dir


REQUIRED_MEMBERS = {
    "calendar.csv",
    "sell_prices.csv",
}
SALES_MEMBERS = {"sales_train_validation.csv", "sales_train_evaluation.csv"}
ZENODO_RECORD = "12636070"


def _validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = {Path(name).name for name in archive.namelist()}
    missing = sorted(REQUIRED_MEMBERS - names)
    if missing:
        raise RuntimeError(f"M5 zip is missing required files: {missing}")
    if not SALES_MEMBERS.intersection(names):
        raise RuntimeError("M5 zip must contain sales_train_validation.csv or sales_train_evaluation.csv")


def download_m5(raw_dir: str | Path | None = None) -> Path:
    root = Path(raw_dir) if raw_dir is not None else case_raw_dir("m5")
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "m5-forecasting-accuracy.zip"
    provenance = {"case": "m5", "attempts": []}
    if zip_path.is_file():
        _validate_zip(zip_path)
        provenance["source"] = "existing"
        write_provenance(root / "provenance.json", provenance)
        return zip_path

    try:
        with urlopen(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60) as response:
            record = json.loads(response.read().decode("utf-8"))
        files = record.get("files", [])
        target = None
        for file_info in files:
            if file_info.get("key") == "m5-forecasting-accuracy.zip":
                links = file_info.get("links", {})
                target = links.get("self") or links.get("download")
                break
        if target is None:
            raise RuntimeError("Zenodo record did not list m5-forecasting-accuracy.zip")
        download_url(target, zip_path, attempts=3, timeout=120)
        _validate_zip(zip_path)
        provenance["source"] = "zenodo"
        write_provenance(root / "provenance.json", provenance)
        return zip_path
    except Exception as exc:
        provenance["attempts"].append({"source": "zenodo", "error": str(exc)})

    result = run_command(
        ["kaggle", "competitions", "download", "-c", "m5-forecasting-accuracy", "-p", str(root)]
    )
    if result.returncode == 0 and zip_path.is_file():
        _validate_zip(zip_path)
        provenance["source"] = "kaggle"
        write_provenance(root / "provenance.json", provenance)
        return zip_path

    provenance["attempts"].append(
        {"source": "kaggle", "returncode": result.returncode, "stderr": result.stderr}
    )
    write_provenance(root / "provenance.json", provenance)
    raise RuntimeError(
        "Could not download M5 automatically. Accept the Kaggle M5 competition rules once "
        "in a browser, install/configure the Kaggle CLI, place credentials in "
        "~/.kaggle/kaggle.json, then rerun python scripts/real_data/run_m5.py."
    )
