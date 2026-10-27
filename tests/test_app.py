from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_exception():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert len(app.metric) == 6
