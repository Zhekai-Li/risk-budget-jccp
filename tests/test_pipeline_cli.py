from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "pipeline" / "reproduce.py"


def test_reproduce_dry_run_lists_all_stages_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-root", str(output), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not output.exists()
    for stage in (
        "preflight",
        "synthetic-service",
        "synthetic-cross-domain",
        "synthetic-capacity",
        "real-data-main",
        "sensitivity",
        "paper-figures",
        "validate",
    ):
        assert f"[{stage}]" in completed.stdout


def test_reproduce_refuses_nonempty_output_without_resume(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-root", str(output), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "without --resume" in completed.stderr
    assert marker.read_text(encoding="utf-8") == "user data"


def test_reproduce_allows_nonempty_output_with_resume_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-root", str(output), "--dry-run", "--resume"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
