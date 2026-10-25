"""Sanity-check the environment: Python version, every library imports, config loads.

Run from the project root:
    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata

# (import name, friendly name) — import names differ from pip names for some
LIBRARIES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("sklift", "scikit-uplift"),
    ("causalml", "causalml"),
    ("econml", "econml"),
    ("shap", "shap"),
    ("matplotlib", "matplotlib"),
    ("plotly", "plotly"),
    ("mlflow", "mlflow"),
    ("streamlit", "streamlit"),
    ("yaml", "pyyaml"),
]


def check_python() -> bool:
    """Warn if not on the Python versions causalml/econml support well."""
    major, minor = sys.version_info[:2]
    ok = (major, minor) in {(3, 10), (3, 11), (3, 12)}
    status = "OK " if ok else "WARN"
    print(f"[{status}] Python {platform.python_version()} ({platform.architecture()[0]})")
    if not ok:
        print("       -> Use Python 3.11: py -3.11 -m venv .venv")
    return ok


def check_imports() -> bool:
    """Try importing each library and print its version."""
    all_ok = True
    for module_name, friendly in LIBRARIES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None) or metadata.version(friendly)
            print(f"[OK ] {friendly:<14} {version}")
        except Exception as exc:  # noqa: BLE001 — we want to see any failure
            all_ok = False
            print(f"[FAIL] {friendly:<14} {type(exc).__name__}: {exc}")
    return all_ok


def check_config() -> bool:
    """Load config.yaml through the project package."""
    try:
        from upliftlab.config import load_config

        cfg = load_config()
        print(f"[OK ] config loaded: project={cfg.project.name}, seed={cfg.project.seed}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] config: {type(exc).__name__}: {exc}")
        print("       -> Did you run `pip install -e .` from the project root?")
        return False


def main() -> int:
    results = [check_python(), check_imports(), check_config()]
    print("\nAll good — environment ready." if all(results) else "\nFix the FAIL lines above.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
