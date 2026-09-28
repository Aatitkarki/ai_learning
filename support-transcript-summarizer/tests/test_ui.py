from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import TRANSCRIPT

UI_PATH = Path(__file__).resolve().parents[1] / "ui.py"


def click(app, label):
    return next(button for button in app.button if button.label == label).click().run()


def test_ui_create_evidence_edit_reopen_export_and_delete(client):
    class ClientContext:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return client

        def __exit__(self, *_args):
            pass

    with patch("httpx.Client", ClientContext):
        app = AppTest.from_file(str(UI_PATH), default_timeout=20).run()
        assert not app.exception
        app.text_area(key="transcript").set_value(TRANSCRIPT)
        app.text_input(key="title").set_value("UI test handover")
        click(app, "Create handover →")
        assert not app.exception
        assert app.session_state["record"]["title"] == "UI test handover"
        app.button(key="issue").click().run()
        assert app.session_state["evidence_lines"] == [1]
        next(area for area in app.text_area if area.label == "Editable handover").set_value(
            "Reviewed draft with L1 evidence."
        )
        click(app, "Save edits")
        assert not app.exception
        assert app.session_state["record"]["is_edited"] is True
        assert app.session_state["record"]["draft_markdown"] == "Reviewed draft with L1 evidence."
        record_id = app.session_state["record"]["id"]
        click(app, "＋ New conversation")
        app.button(key=record_id).click().run()
        assert app.session_state["record"]["id"] == record_id
        assert len(app.get("download_button")) == 2
        app.checkbox(key=f"delete-{record_id}").check().run()
        click(app, "Delete permanently")
        assert not app.exception
        assert client.get("/api/summaries").json() == []


def test_ui_empty_input_and_example_loading(client):
    class ClientContext:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return client

        def __exit__(self, *_args):
            pass

    with patch("httpx.Client", ClientContext):
        app = AppTest.from_file(str(UI_PATH), default_timeout=20).run()
        click(app, "Create handover →")
        assert any("Paste a conversation" in error.value for error in app.error)
        app.selectbox(key="sample").select("Incomplete Conversation").run()
        assert "My transfer still has not arrived" in app.text_area(key="transcript").value
        assert not app.exception


def test_ui_api_offline_is_actionable():
    import httpx

    with patch("httpx.Client.request", side_effect=httpx.ConnectError("offline")):
        app = AppTest.from_file(str(UI_PATH), default_timeout=20).run()
    assert not app.exception
    assert any("Cannot reach the support API" in error.value for error in app.error)
