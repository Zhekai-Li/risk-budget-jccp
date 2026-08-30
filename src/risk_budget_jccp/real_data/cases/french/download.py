from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urljoin
from urllib.request import urlopen

from risk_budget_jccp.real_data.common.downloading import download_url, write_provenance
from risk_budget_jccp.real_data.common.paths import case_raw_dir


LIBRARY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"


def download_french(raw_dir: str | Path | None = None) -> Path:
    root = Path(raw_dir) if raw_dir is not None else case_raw_dir("french")
    root.mkdir(parents=True, exist_ok=True)
    output = root / "49_Industry_Portfolios_Daily_CSV.zip"
    provenance = {"case": "french", "attempts": []}
    if output.is_file():
        provenance["source"] = "existing"
        write_provenance(root / "provenance.json", provenance)
        return output
    try:
        with urlopen(LIBRARY_URL, timeout=60) as response:
            html = response.read().decode("latin1", errors="replace")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
        target = None
        for href in hrefs:
            low = href.lower()
            if "49" in low and "industry" in low and "daily" in low and "csv" in low and low.endswith(".zip"):
                target = urljoin(LIBRARY_URL, href)
                break
        if target is None:
            raise RuntimeError("could not find 49 Industry Portfolios daily CSV zip link")
        download_url(target, output, attempts=4, timeout=90)
        provenance["source"] = "kenneth_french_direct"
        provenance["url"] = target
        write_provenance(root / "provenance.json", provenance)
        return output
    except Exception as exc:
        provenance["attempts"].append({"source": "kenneth_french_direct", "error": str(exc)})

    try:
        from pandas_datareader import data as pdr_data
        frame = pdr_data.DataReader("49_Industry_Portfolios_Daily", "famafrench")[0]
        frame.to_csv(root / "49_Industry_Portfolios_Daily_pandas_datareader.csv")
        provenance["source"] = "pandas_datareader"
        write_provenance(root / "provenance.json", provenance)
        return root / "49_Industry_Portfolios_Daily_pandas_datareader.csv"
    except Exception as exc:
        provenance["attempts"].append({"source": "pandas_datareader", "error": str(exc)})
        write_provenance(root / "provenance.json", provenance)
        raise RuntimeError(
            "Could not download Kenneth French daily industry data. The Dartmouth site may be "
            "temporarily unavailable; install pandas_datareader for fallback or place the daily "
            "49 Industry Portfolios CSV/ZIP in data/raw/real_data/french and rerun."
        ) from exc
