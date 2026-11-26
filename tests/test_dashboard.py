"""End-to-end rendering test for the Streamlit dashboard."""

import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODEL_PIPELINE_PATH, PROJECT_ROOT


@pytest.mark.skipif(
    not MODEL_PIPELINE_PATH.exists(),
    reason="Trained model artifacts not generated yet; run train.py first.",
)
def test_streamlit_dashboard_renders_without_errors():
    dashboard_path = PROJECT_ROOT / "app" / "dashboard.py"
    at = AppTest.from_file(str(dashboard_path), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.metric) == 3
    assert len(at.tabs) == 3
