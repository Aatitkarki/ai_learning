from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest


def test_all_workbench_pages_render_without_a_model(indexed, mock_http):
    lab, index, docs = indexed

    def handler(request):
        payloads = {
            "/api/health": lab.catalog(),
            "/api/documents": docs,
            "/api/indexes": [index],
            "/api/evaluations": [],
            "/api/runs": [],
            "/api/conversations": [],
        }
        assert request.url.path in payloads, request.url.path
        return httpx.Response(200, json=payloads[request.url.path])

    mock_http(handler)
    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "ui.py"))
    app.run(timeout=15)
    assert not app.exception
    next(widget for widget in app.radio if widget.label == "Answer mode").set_value(
        "openai_compatible"
    ).run(timeout=15)
    assert not app.exception
    assert any("API setup pending" in item.value for item in app.info)
    for page in ["Documents", "Index studio", "Compare", "Evaluations", "Runs & reviews"]:
        app.sidebar.radio[0].set_value(page).run(timeout=15)
        assert not app.exception, page
        if page == "Index studio":
            next(
                widget for widget in app.selectbox if widget.label == "Embedding provider"
            ).set_value("openai_compatible").run(timeout=15)
            assert not app.exception
