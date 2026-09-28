"""Streamlit client: every durable operation goes through the FastAPI service."""

import html
import json
from pathlib import Path

import httpx
import streamlit as st

from summarizer.config import Settings
from summarizer.models import MAX_TRANSCRIPT_CHARS, MAX_UPLOAD_BYTES, SavedSummary
from summarizer.rendering import export_markdown

st.set_page_config(
    page_title="Handover · Support Transcript Summarizer", page_icon="📝", layout="wide"
)
settings = Settings()


def api(method, path, **kwargs):
    try:
        with httpx.Client(
            base_url=settings.api_base_url,
            trust_env=False,
            timeout=settings.model_timeout_seconds + 15,
        ) as client:
            response = client.request(method, path, **kwargs)
        if response.is_error:
            body = response.json()
            error = body.get("error", {})
            message = error.get("message", body.get("detail", "Request failed."))
            details = error.get("details", [])
            st.error(str(message))
            for detail in details:
                st.caption(f"{detail['field']}: {detail['message']}")
            return None
        return True if response.status_code == 204 else response.json()
    except httpx.TimeoutException:
        st.error("The request timed out. Check History before retrying; it may have completed.")
    except (httpx.HTTPError, ValueError):
        st.error("Cannot reach the support API. Start it, then try again.")
    return None


def select_record(record):
    st.session_state.record = record
    st.session_state.evidence_lines = []


def load_sample():
    sample = st.session_state.sample
    if sample != "Choose a sample…":
        st.session_state.transcript = samples[sample]
        st.session_state.title = sample


sample_dir = Path(__file__).parent / "examples"
samples = {
    path.stem.replace("_", " ").title(): path.read_text()
    for path in sorted(sample_dir.glob("*.txt"))
}
with st.sidebar:
    st.title("◈ Handover")
    st.caption("Support Transcript Summarizer")
    st.divider()
    st.subheader("Saved handovers")
    if st.button("＋ New conversation", width="stretch"):
        st.session_state.pop("record", None)
        st.rerun()
    page = st.number_input("History page", min_value=1, step=1, value=1)
    history = api("GET", "/api/summaries", params={"limit": 10, "offset": (page - 1) * 10})
    if history is not None:
        if not history:
            st.caption("No handovers on this page.")
        for item in history:
            label = item["title"] + (" · edited" if item["is_edited"] else "")
            if st.button(label, key=item["id"], width="stretch"):
                record = api("GET", f"/api/summaries/{item['id']}")
                if record:
                    select_record(record)
                    st.rerun()
    st.divider()
    st.caption("Local model · Local history")
    st.caption("Portfolio demo. Use fictional conversations.")

st.caption("CONVERSATION → CLARITY")
st.title("A clear handover. Grounded in the conversation.")
st.write(
    "Capture the issue, keep the important details, and see exactly where each fact came from."
)

if "record" not in st.session_state:
    status = api("GET", "/health")
    if status:
        if status["model_status"]["available"]:
            st.success(f"Local model ready · {status['model']}", icon="✅")
        else:
            st.warning(status["model_status"]["message"] + " See the README's model setup steps.")
    left, right = st.columns([3, 1], gap="large")
    with left:
        st.subheader("Start with a conversation")
        st.selectbox(
            "Try a fictional example",
            ["Choose a sample…", *samples],
            key="sample",
            on_change=load_sample,
        )
        st.session_state.setdefault("title", "Support conversation")
        st.text_input("Handover title", key="title", max_chars=100)
        mode = st.radio("Input method", ["Paste text", "Upload .txt"], horizontal=True)
        uploaded = None
        if mode == "Paste text":
            text = st.text_area(
                "Transcript",
                height=330,
                key="transcript",
                placeholder="Customer: My transfer hasn't arrived.\nAgent: …",
            )
            st.caption(
                f"{len(text):,} / {MAX_TRANSCRIPT_CHARS:,} characters · "
                "Maximum 80 lines and 8,000 UTF-8 bytes"
            )
        else:
            uploaded = st.file_uploader("UTF-8 transcript", type=["txt"])
            st.caption(
                "Maximum file size: 24,000 bytes. Text: 6,000 characters, "
                "80 lines and 8,000 UTF-8 bytes."
            )
            text = ""
        if st.button("Create handover →", type="primary", width="stretch"):
            if mode == "Paste text" and not text.strip():
                st.error("Paste a conversation first.")
            elif mode == "Upload .txt" and uploaded is None:
                st.error("Choose a .txt file first.")
            elif uploaded is not None and uploaded.size > MAX_UPLOAD_BYTES:
                st.error("File exceeds 24,000 bytes.")
            else:
                with st.spinner("Reading the conversation and checking evidence…"):
                    if uploaded is not None:
                        record = api(
                            "POST",
                            "/api/summaries/upload",
                            files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")},
                            data={"title": st.session_state.title},
                        )
                    else:
                        record = api(
                            "POST",
                            "/api/summaries",
                            json={"transcript": text, "title": st.session_state.title},
                        )
                if record:
                    select_record(record)
                    st.rerun()
    with right:
        st.subheader("Made for the next person")
        st.write(
            "**01 · Preserve the facts**\n\n"
            "Names, references and amounts stay tied to their source."
        )
        st.write(
            "**02 · Make gaps visible**\n\nMissing details stay unknown. Conflicts stay visible."
        )
        st.write(
            "**03 · Review, then use**\n\n"
            "Check supporting lines and edit your handover before sharing."
        )
        st.info(
            "Suggested next steps are proposals. They are never sent or carried out by this app."
        )
else:
    record = st.session_state.record
    summary = record["summary"]
    st.subheader(record["title"])
    metrics = record["metrics"]
    st.caption(
        f"{metrics['model']} · {metrics['latency_ms'] / 1000:.1f}s · "
        f"Tokens: {metrics['input_tokens']} in / {metrics['output_tokens']} out"
    )
    st.info(
        "Evidence references and quoted values were checked. Review meaning and attribution "
        "against the conversation; these checks cannot prove factual accuracy."
    )
    left, right = st.columns([3, 2], gap="large")

    def show_claim(text, refs, key):
        st.text(text)
        if refs and st.button("Inspect " + ", ".join(f"L{x}" for x in refs), key=key):
            st.session_state.evidence_lines = refs

    with left:
        structured, edit = st.tabs(["Evidence-linked summary", "Edit & export"])
        with structured:
            st.subheader("Customer issue")
            issue = summary["customer_issue"]
            if issue:
                show_claim(issue["text"], issue["source_lines"], "issue")
            else:
                st.write("Unknown / not stated")
            st.subheader("Key details")
            for field, label in [
                ("customer_name", "Customer"),
                ("amount", "Amount"),
                ("transfer_reference", "Reference"),
                ("stated_date", "Date as stated"),
                ("delay_reason", "Delay reason"),
            ]:
                fact = summary[field]
                st.caption(label.upper())
                show_claim(
                    fact["value"] or "Unknown / not established", fact["source_lines"], field
                )
            for field, label in [
                ("key_facts", "Other key facts"),
                ("actions_taken", "Actions already taken"),
                ("contradictions", "Conflicting statements"),
            ]:
                st.subheader(label)
                if not summary[field]:
                    st.caption("None stated.")
                for i, claim in enumerate(summary[field]):
                    show_claim(claim["text"], claim["source_lines"], f"{field}-{i}")
            st.subheader("Unresolved questions")
            for question in summary["unresolved_questions"]:
                st.text("• " + question)
            if not summary["unresolved_questions"]:
                st.caption("None identified.")
            st.subheader("Suggested next steps")
            st.caption("PROPOSALS · NOT COMPLETED ACTIONS")
            for step in summary["suggested_next_steps"]:
                st.text("• " + step)
            if not summary["suggested_next_steps"]:
                st.caption("None suggested.")
        with edit:
            st.caption(
                "Edit the handover in Markdown. The original structured summary and its "
                "evidence remain available in the other tab. Edits are not evidence-checked."
            )
            with st.form(f"draft-{record['id']}-{record['revision']}"):
                draft = st.text_area(
                    "Editable handover",
                    value=record["draft_markdown"],
                    height=560,
                    max_chars=20_000,
                )
                if st.form_submit_button("Save edits", type="primary"):
                    updated = api(
                        "PATCH",
                        f"/api/summaries/{record['id']}",
                        json={"draft_markdown": draft, "revision": record["revision"]},
                    )
                    if updated:
                        select_record(updated)
                        st.rerun()
            st.download_button(
                "Download saved handover",
                export_markdown(SavedSummary.model_validate(record)),
                "support-handover.md",
                "text/markdown",
            )
            st.download_button(
                "Download original structured JSON",
                json.dumps(summary, indent=2, ensure_ascii=False),
                "summary.json",
                "application/json",
            )
            st.caption("Downloads contain the saved version. Save any edits first.")
            if record["is_edited"]:
                st.warning(
                    "This saved draft contains human edits that have not been evidence-checked."
                )
        with st.expander("Delete this handover"):
            confirm = st.checkbox(
                "Delete the handover and source transcript from local history",
                key=f"delete-{record['id']}",
            )
            if st.button("Delete permanently", disabled=not confirm):
                if api("DELETE", f"/api/summaries/{record['id']}"):
                    st.session_state.pop("record", None)
                    st.rerun()
    with right:
        st.subheader("Source conversation")
        selected = st.session_state.get("evidence_lines", [])
        st.caption("Select an Inspect button to highlight supporting lines.")
        if selected:
            st.success("Selected evidence: " + ", ".join(f"L{x}" for x in selected))
            for i in selected:
                st.text(f"L{i}  {record['transcript'].split(chr(10))[i - 1]}")
        with st.container(height=670):
            for i, line in enumerate(record["transcript"].split("\n"), 1):
                color = "#D4EEE5" if i in selected else "#FFFFFF"
                st.markdown(
                    f'<div id="line-{i}" style="padding:12px;margin:6px 0;'
                    f'border:1px solid #DCE5E1;border-radius:8px;background:{color}">'
                    f"<small>L{i}</small><br>{html.escape(line) or '&nbsp;'}</div>",
                    unsafe_allow_html=True,
                )
