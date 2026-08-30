from __future__ import annotations

from pathlib import Path

from risk_budget_jccp.real_data.common.downloading import require_api_key, run_command, write_provenance
from risk_budget_jccp.real_data.common.paths import case_raw_dir


RTS_REPO = "https://github.com/GridMod/RTS-GMLC.git"


def download_power(raw_dir: str | Path | None = None, *, use_external_api: bool, fallback_to_builtin: bool) -> Path:
    root = Path(raw_dir) if raw_dir is not None else case_raw_dir("power")
    root.mkdir(parents=True, exist_ok=True)
    repo_dir = root / "RTS-GMLC"
    provenance = {"case": "power", "external_api_requested": bool(use_external_api), "attempts": []}
    if not repo_dir.is_dir():
        result = run_command(["git", "clone", "--depth", "1", RTS_REPO, str(repo_dir)])
        if result.returncode != 0:
            provenance["attempts"].append({"source": "git", "stderr": result.stderr})
            write_provenance(root / "provenance.json", provenance)
            raise RuntimeError(f"failed to clone RTS-GMLC from {RTS_REPO}: {result.stderr}")
    provenance["rts_gmlc"] = "available"

    if use_external_api:
        api_key = require_api_key("NREL_API_KEY", "NLR_API_KEY")
        if api_key is None:
            if not fallback_to_builtin:
                raise RuntimeError(
                    "External WIND Toolkit/NSRDB access requested, but NREL_API_KEY or "
                    "NLR_API_KEY is not set. Set one of those environment variables or enable "
                    "fallback_to_rts_builtin."
                )
            provenance["external_api"] = "skipped_missing_NREL_API_KEY_or_NLR_API_KEY"
            provenance["renewable_source"] = "rts_gmlc_builtin_timeseries"
        else:
            provenance["external_api"] = (
                "not_downloaded_in_default_mac_profile; endpoints documented for "
                "developer.nlr.gov WIND Toolkit and NSRDB direct CSV mode"
            )
            provenance["renewable_source"] = "rts_gmlc_builtin_timeseries"
    else:
        provenance["renewable_source"] = "rts_gmlc_builtin_timeseries"
    write_provenance(root / "provenance.json", provenance)
    return repo_dir
