"""Tests for the config loader."""

from upliftlab.config import PROJECT_ROOT, load_config


def test_config_loads_core_sections() -> None:
    cfg = load_config()
    assert cfg.project.seed == 42
    assert 0 < cfg.split.test_size < 1
    assert cfg.business.cost_per_offer >= 0


def test_section_to_dict_is_usable_as_kwargs() -> None:
    params = load_config().lightgbm.to_dict()
    assert params["n_estimators"] > 0
    assert "learning_rate" in params


def test_path_resolves_under_project_root() -> None:
    p = load_config().path("figures")
    assert p.is_dir()
    assert PROJECT_ROOT in p.parents
