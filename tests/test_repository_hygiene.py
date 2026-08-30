from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".log", ".md", ".py", ".tex", ".toml", ".txt", ".yaml", ".yml"}
NON_PDF_FIGURE_SUFFIXES = {".eps", ".jpeg", ".jpg", ".png", ".svg", ".tif", ".tiff"}


def _tracked_text_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = []
    for relative in completed.stdout.splitlines():
        path = REPO_ROOT / relative
        if (
            path.suffix.lower() in TEXT_SUFFIXES
            and relative != "tests/test_repository_hygiene.py"
            and path.is_file()
        ):
            paths.append(path)
    return paths


def test_tracked_text_has_no_machine_paths_environment_names_or_credentials() -> None:
    forbidden = {
        "absolute macOS user path": re.compile(r"/Users/[^/\s]+/"),
        "absolute Linux user path": re.compile(r"/home/[^/\s]+/"),
        "local conda environment": re.compile(r"AGICookBook"),
        "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
        "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    }
    failures: list[str] = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in forbidden.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(REPO_ROOT)}: {label}")
    assert not failures, "\n".join(failures)


def test_real_data_has_one_canonical_main_config() -> None:
    config_dir = REPO_ROOT / "configs" / "real_data"
    assert sorted(path.name for path in config_dir.glob("*.yaml")) == ["main.yaml", "sensitivity.yaml"]
    main_digest = hashlib.sha256((config_dir / "main.yaml").read_bytes()).hexdigest()
    duplicates = [
        path.name
        for path in config_dir.glob("*.yaml")
        if path.name != "main.yaml" and hashlib.sha256(path.read_bytes()).hexdigest() == main_digest
    ]
    assert duplicates == []


def test_live_files_do_not_reference_retired_entry_points() -> None:
    retired = (
        "configs/benchmarks",
        "configs/legacy",
        "scripts/benchmarks",
        "scripts/legacy",
        "all_mac.yaml",
        "sensitivity_all.yaml",
        "scripts/make_paper_figures.py",
        "scripts/run_synthetic.py",
        "scripts/run_synthetic_cross_domain.py",
        "scripts/validate_final_results.py",
        "retail_main_results.csv",
        "power_main_results.csv",
    )
    failures = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for value in retired:
            if value in text:
                failures.append(f"{path.relative_to(REPO_ROOT)}: {value}")
    assert not failures, "\n".join(failures)


def test_experiments_have_one_canonical_entry_config_and_result_set() -> None:
    expected_configs = {
        "synthetic_service.yaml",
        "synthetic_cross_domain.yaml",
        "synthetic_coupled_capacity.yaml",
    }
    expected_runners = {
        "run_synthetic.py",
        "run_synthetic_cross_domain.py",
        "run_synthetic_coupled_capacity.py",
    }
    expected_results = {
        "synthetic_service.csv",
        "synthetic_service_raw.csv",
        "synthetic_cross_domain.csv",
        "synthetic_cross_domain_raw.csv",
        "synthetic_coupled_capacity.csv",
    }
    assert {path.name for path in (REPO_ROOT / "configs" / "synthetic").glob("*.yaml")} == expected_configs
    assert {path.name for path in (REPO_ROOT / "scripts" / "synthetic").glob("*.py")} == expected_runners
    assert {path.name for path in (REPO_ROOT / "results" / "tables").iterdir()} == expected_results
    assert not (REPO_ROOT / "configs" / "legacy").exists()
    assert not (REPO_ROOT / "scripts" / "legacy").exists()
    assert not (REPO_ROOT / "docs" / "superpowers").exists()


def test_tracked_text_has_no_merge_conflict_markers() -> None:
    markers = ("<<<<<<<", "=======", ">>>>>>>")
    failures = []
    for path in _tracked_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in markers):
            failures.append(path.relative_to(REPO_ROOT).as_posix())
    assert not failures, "\n".join(failures)


def test_reference_figures_are_pdf_only() -> None:
    non_pdf_figures = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "results").rglob("*")
        if path.is_file() and path.suffix.lower() in NON_PDF_FIGURE_SUFFIXES
    )
    assert non_pdf_figures == []
    assert ".png" not in (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()


def test_readme_local_links_resolve() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\((?!https?://)([^)#]+)(?:#[^)]+)?\)", text)
    missing = sorted(target for target in targets if not (REPO_ROOT / target).exists())
    assert missing == []
