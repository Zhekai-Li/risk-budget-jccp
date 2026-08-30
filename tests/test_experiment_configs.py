from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config(name: str) -> dict[str, object]:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "synthetic" / name).read_text(encoding="utf-8")
    )


def test_synthetic_configs_archive_thirty_replications() -> None:
    service = _load_config("synthetic_service.yaml")
    cross_domain = _load_config("synthetic_cross_domain.yaml")
    assert int(service["replications"]) == 30
    assert int(cross_domain["replications"]) == 30
    assert service["methods"] == ["bernstein", "cantelli", "cvar"]
    assert len(cross_domain["domains"]) == 5


def test_coupled_config_uses_one_fixed_seed() -> None:
    coupled = _load_config("synthetic_coupled_capacity.yaml")
    assert coupled["seed"] == 7
    assert "base_seed" not in coupled
    assert "replications" not in coupled
