"""Load config/config.yaml into a typed, read-only object.

Usage:
    from upliftlab.config import load_config
    cfg = load_config()
    cfg.project.seed            # 42
    cfg.path("figures")         # absolute Path to reports/figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# src/upliftlab/config.py -> parents[2] is the project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Section:
    """A config section whose keys are readable as attributes (cfg.split.test_size)."""

    _data: dict[str, Any]

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{key}'") from exc
        return Section(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict[str, Any]:
        """Plain dict copy, e.g. to pass as **kwargs to LightGBM or log to MLflow."""
        return dict(self._data)


@dataclass(frozen=True)
class Config:
    """Whole project config. Top-level sections are exposed as attributes."""

    raw: dict[str, Any]
    root: Path = PROJECT_ROOT

    def __getattr__(self, key: str) -> Any:
        if key in self.raw:
            value = self.raw[key]
            return Section(value) if isinstance(value, dict) else value
        raise AttributeError(f"Config has no section '{key}'")

    def path(self, name: str, create: bool = True) -> Path:
        """Absolute path for an entry under `paths:`; creates the folder by default."""
        p = self.root / self.raw["paths"][name]
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: Path | str | None = None) -> Config:
    """Read the YAML config file and return a Config object."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
