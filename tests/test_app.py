"""Smoke-test the Streamlit app headlessly (runs every tab's code without a browser)."""

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
ARTIFACTS = APP.parent / "artifacts" / "metadata.json"


@pytest.mark.skipif(not ARTIFACTS.exists(), reason="run scripts/run_pipeline.py first")
def test_app_runs_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    messages = [m.value for m in list(at.success) + list(at.error)]
    assert any("SEND" in m or "DO NOT" in m for m in messages)


@pytest.mark.skipif(not ARTIFACTS.exists(), reason="run scripts/run_pipeline.py first")
def test_app_reacts_to_expensive_offers() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.slider[0].set_value(5.0).run()  # $5 per offer
    assert not at.exception
    assert any("DO NOT" in e.value for e in at.error)  # no customer is worth $5 here
