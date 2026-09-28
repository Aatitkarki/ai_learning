"""Streamlit workbench. All persistent operations go through the FastAPI service."""

import json

import httpx
import streamlit as st

from evaluation.runner import markdown_report
from knowledge_lab.config import Settings

settings = Settings()
API = settings.api_url
st.set_page_config(page_title="Knowledge Lab", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top: 2rem; max-width: 1450px;}
h1 {letter-spacing: -0.05em; font-weight: 750 !important;}
h2, h3 {letter-spacing: -0.025em;}
[data-testid="stMetric"] {background: #172033; padding: 1rem; border-radius: 12px;}
[data-testid="stSidebar"] {border-right: 1px solid #27344a;}
.eyebrow {color: #51d6ba; font-size: .75rem; letter-spacing: .18em; font-weight: 700;}
.intro {color: #a7b7cd; font-size: 1.05rem; margin-bottom: 1.3rem;}
</style>""",
    unsafe_allow_html=True,
)


def api(method, path, **kwargs):
    try:
        with httpx.Client(base_url=API, timeout=180, trust_env=False) as client:
            response = client.request(method, f"/api{path}", **kwargs)
        if response.is_error:
            detail = response.json().get("detail", response.text)
            st.error(detail if isinstance(detail, str) else json.dumps(detail))
            return None
        return response.json() if response.content else True
    except (httpx.HTTPError, ValueError):
        st.error("The local API is unavailable. Start it with: python scripts/dev.py")
        return None


def title(eyebrow, heading, description):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(heading)
    st.markdown(f'<div class="intro">{description}</div>', unsafe_allow_html=True)


def index_label(index):
    config = index["config"]
    return f"{config['name']} · {config['store']} · {index['id'][:6]}"


def configuration(prefix, indexes, documents, compact=False):
    selected = st.selectbox("Index", indexes, format_func=index_label, key=f"{prefix}_index")
    method = st.selectbox(
        "Retrieval method",
        ["bm25", "vector", "hybrid", "mmr"],
        index=2,
        key=f"{prefix}_method",
        help="BM25: keywords. Vector: cosine similarity. Hybrid: weighted rank "
        "fusion. MMR: relevant passages with less repetition.",
    )
    k = st.slider("Passages to retrieve", 1, 20, 5, key=f"{prefix}_k")
    with st.expander("Retrieval controls", expanded=False):
        candidate_k = st.slider("Candidate pool", max(20, k), 200, 30, key=f"{prefix}_candidates")
        weight = st.slider(
            "Vector weight in hybrid search", 0.0, 1.0, 0.5, 0.05, key=f"{prefix}_weight"
        )
        diversity = st.slider(
            "MMR relevance weight", 0.0, 1.0, 0.65, 0.05, key=f"{prefix}_diversity"
        )
        threshold = st.slider(
            "Minimum cosine score", 0.0, 1.0, 0.15, 0.01, key=f"{prefix}_threshold"
        )
        bm25_min = st.number_input(
            "Minimum BM25 score", min_value=0.0, value=0.1, step=0.1, key=f"{prefix}_bm25"
        )
        rerank = st.selectbox("Reranking", ["none", "lexical"], key=f"{prefix}_rerank")
        expand = st.checkbox(
            "Expand common remittance terms",
            key=f"{prefix}_expand",
            help="Uses a fixed, inspectable synonym dictionary. No model call.",
        )
        available = [d for d in documents if d["id"] in selected["config"]["document_ids"]]
        filters = st.multiselect(
            "Restrict documents (empty = all)",
            available,
            format_func=lambda d: d["filename"],
            key=f"{prefix}_filters",
        )
    return {
        "index_id": selected["id"],
        "method": method,
        "top_k": k,
        "candidate_k": candidate_k,
        "vector_weight": weight,
        "mmr_lambda": diversity,
        "min_vector_score": threshold,
        "min_bm25_score": bm25_min,
        "rerank": rerank,
        "expand_query": expand,
        "document_ids": [d["id"] for d in filters],
    }


def show_sources(sources):
    for source in sources:
        label = source.get("label", "Passage")
        with st.expander(f"{label} · {source['filename']} · page {source['page']}"):
            if source.get("heading"):
                st.caption(source["heading"])
            st.text(source["text"])
            st.caption(f"Characters {source['start']}–{source['end']} · {source['id'][:8]}")
            st.link_button(
                "Open original source",
                f"{API}/api/documents/{source['document_id']}/source#page={source['page']}",
            )


def show_retrieval(result):
    cols = st.columns(2)
    cols[0].metric("Passages", len(result["hits"]))
    cols[1].metric("Retrieval", f"{result['timings_ms']['total']:.1f} ms")
    for warning in result["warnings"]:
        st.caption(warning)
    st.caption(f"Search query: {result['expanded_query']}")
    for rank, hit in enumerate(result["hits"], 1):
        chunk = hit["chunk"]
        with st.expander(
            f"{rank:02} · {chunk['filename']} · score {hit['score']:.4f}", expanded=rank == 1
        ):
            st.caption(f"Page {chunk['page']} · {chunk['heading'] or 'Text passage'}")
            st.text(chunk["text"])
            st.json(hit["scores"], expanded=False)
            st.link_button(
                "Open source",
                f"{API}/api/documents/{chunk['document_id']}/source#page={chunk['page']}",
            )
    if not result["hits"]:
        st.info("No passages cleared these retrieval thresholds.")
    with st.expander("Trace & configuration"):
        st.json({k: v for k, v in result.items() if k != "hits"})


def ask(indexes, documents, health):
    title(
        "RETRIEVE / READ / VERIFY",
        "Ask your documents.",
        "Follow the evidence from a question to its source. Tune each step as you learn.",
    )
    if not indexes:
        st.info(
            "Start in Documents: load the fictional policy pack or upload your files. "
            "Then create an index in Index studio."
        )
        return
    left, right = st.columns([1, 2], gap="large")
    with left:
        resumed = st.session_state.pop("resume_index", None)
        if resumed:
            matching = next((index for index in indexes if index["id"] == resumed), None)
            if matching:
                st.session_state["ask_index"] = matching
        config = configuration("ask", indexes, documents)
        mode = st.radio(
            "Answer mode",
            ["evidence", "openai_compatible", "ollama"],
            format_func=lambda m: {
                "evidence": "Evidence only · no model",
                "openai_compatible": "Generate with configured API",
                "ollama": "Generate with local Ollama",
            }[m],
        )
        if mode == "ollama" and not health["chat_model"]:
            st.warning("No chat model configured. Set LAB_CHAT_MODEL later and restart the API.")
        if mode == "openai_compatible":
            connection = health["openai"]
            if not connection["base_url_configured"] or not connection["chat_model"]:
                st.info(
                    "API setup pending. Add the URL, key, and chat model to .env later, "
                    "then restart the app."
                )
            st.caption("This mode sends the question and selected passages to your configured API.")
        use_history = st.checkbox("Use previous question for follow-ups", value=True)
        if st.button("New conversation", use_container_width=True):
            st.session_state.pop("conversation_id", None)
            st.session_state.pop("answer", None)
        if st.session_state.get("conversation_index") != config["index_id"]:
            st.session_state.pop("conversation_id", None)
            st.session_state.pop("answer", None)
            st.session_state.pop("search_result", None)
        st.session_state.conversation_index = config["index_id"]
        with st.expander("Generation controls"):
            temperature = st.slider(
                "Temperature",
                0.0,
                1.0,
                0.0,
                0.05,
                disabled=mode == "openai_compatible" and not health["openai"]["send_temperature"],
                help="API temperature is sent only when LAB_OPENAI_SEND_TEMPERATURE=true.",
            )
            max_tokens = st.slider("Maximum output tokens", 64, 4096, 1024, 64)
            context_chars = st.slider("Evidence character budget", 1000, 32000, 14000, 1000)
    with right:
        question = st.text_area(
            "Question",
            value="What is the cancellation refund timeline?",
            height=100,
            max_chars=2000,
        )
        retrieve, answer = st.columns(2)
        if retrieve.button("Inspect retrieval", use_container_width=True):
            result = api("POST", "/search", json={"question": question, "config": config})
            if result:
                st.session_state.search_result = result
                st.session_state.pop("answer", None)
        if answer.button("Ask with evidence", type="primary", use_container_width=True):
            placeholder = st.empty()
            draft = ""
            st.session_state.pop("answer", None)
            try:
                with (
                    httpx.Client(timeout=180, trust_env=False) as client,
                    client.stream(
                        "POST",
                        f"{API}/api/chat",
                        json={
                            "question": question,
                            "config": config,
                            "mode": mode,
                            "conversation_id": st.session_state.get("conversation_id"),
                            "use_followup_context": use_history,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "context_chars": context_chars,
                        },
                    ) as response,
                ):
                    response.raise_for_status()
                    completed = False
                    for line in response.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event["type"] == "token":
                            draft += event["text"]
                            placeholder.text("Draft — awaiting citation checks\n\n" + draft)
                        elif event["type"] == "retrieval":
                            st.session_state.search_result = event["result"]
                        elif event["type"] == "complete":
                            st.session_state.answer = event["result"]
                            st.session_state.conversation_id = event["result"]["conversation_id"]
                            completed = True
                        elif event["type"] == "error":
                            st.error(event["message"])
                    if not completed:
                        st.warning("No completed answer was saved.")
            except (httpx.HTTPError, ValueError):
                st.error("The answer stream failed. Check the API and configured local model.")
            placeholder.empty()
        result = st.session_state.get("answer")
        if result:
            st.subheader("Evidence notebook" if result["mode"] == "evidence" else "Answer")
            st.caption(
                f"{result['status'].replace('_', ' ').title()} · "
                f"{result['total_ms']:.0f} ms · saved run {result['id'][:8]}"
            )
            st.text(result["answer"])
            if result.get("warning"):
                st.warning(result["warning"])
            show_sources(result["sources"])
        if st.session_state.get("search_result"):
            st.subheader("Retrieval inspector")
            show_retrieval(st.session_state.search_result)


def library(documents):
    title(
        "YOUR LOCAL CORPUS",
        "Documents",
        "Import, inspect, and manage the evidence behind every answer.",
    )
    with st.container(border=True):
        upload_col, sample_col = st.columns([2, 1], gap="large")
        with upload_col:
            files = st.file_uploader(
                "Add source documents",
                type=["md", "markdown", "txt", "pdf"],
                accept_multiple_files=True,
            )
            st.caption("UTF-8 text and text-based PDFs · up to 10 MB per document · no OCR")
            if st.button("Import selected files", disabled=not files, type="primary"):
                all_imported = True
                for file in files:
                    result = api("POST", "/documents", files={"file": (file.name, file.getvalue())})
                    all_imported = all_imported and bool(result)
                    if result:
                        st.success(
                            f"{'Already imported' if result.get('duplicate') else 'Imported'}: "
                            f"{result['filename']}"
                        )
                if all_imported:
                    st.rerun()
        with sample_col:
            st.markdown("**Start with a controlled corpus**")
            st.write(
                "12 fictional remittance policies, including archived rules and an untrusted "
                "attachment. Paired with 40 evaluation questions."
            )
            if st.button("Load demo policies", use_container_width=True):
                if api("POST", "/documents/seed"):
                    st.rerun()
    if not documents:
        st.info("Your library is empty. Add documents above to start an experiment.")
        return
    cols = st.columns(3)
    cols[0].metric("Documents", len(documents))
    cols[1].metric("Pages", sum(d["page_count"] for d in documents))
    cols[2].metric("Characters", f"{sum(d['characters'] for d in documents):,}")
    for doc in documents:
        with st.expander(
            f"{doc['filename']} · {doc['page_count']} page(s) · {doc['characters']:,} chars"
        ):
            st.caption(f"SHA-256 {doc['sha256']}")
            for warning in doc.get("warnings", []):
                st.warning(warning)
            if st.button("Inspect extracted text", key=f"inspect_{doc['id']}"):
                detail = api("GET", f"/documents/{doc['id']}")
                if detail:
                    for page in detail["pages"]:
                        st.caption(f"Page {page['number']}")
                        st.text(page["text"])
            confirm = st.checkbox(
                "Delete this document, its indexes, and related saved runs",
                key=f"delete_confirm_{doc['id']}",
            )
            if st.button("Delete document", key=f"delete_{doc['id']}", disabled=not confirm):
                if api("DELETE", f"/documents/{doc['id']}"):
                    st.session_state.pop("answer", None)
                    st.session_state.pop("search_result", None)
                    st.rerun()


def studio(indexes, documents, health):
    title(
        "CHUNKS → EMBEDDINGS → STORE",
        "Index studio",
        "Create independent snapshots. Change one variable, then compare.",
    )
    build, inspect = st.columns([1, 1.3], gap="large")
    with build:
        st.subheader("Build an index")
        name = st.text_input("Experiment name", value="Policy baseline", max_chars=80)
        selected = st.multiselect(
            "Source documents", documents, default=documents, format_func=lambda d: d["filename"]
        )
        store = st.selectbox("Vector backend", health["stores"])
        st.caption(
            {
                "sqlite": "Built-in SQLite persistence + exact NumPy cosine search.",
                "pgvector": "PostgreSQL with pgvector. Requires its extra and a running database.",
                "chroma": "Embedded Chroma with cosine HNSW. Requires the chroma extra.",
                "qdrant": "Local Qdrant service via REST. No embedding library required.",
                "faiss": "Local FAISS index files. Requires the faiss extra.",
            }[store]
        )
        embedding = st.selectbox(
            "Embedding provider",
            ["hash", "openai_compatible", "ollama"],
            format_func=lambda e: {
                "hash": "Deterministic hash · no model",
                "openai_compatible": "Configured API embeddings",
                "ollama": "Local Ollama embeddings",
            }[e],
        )
        model = st.text_input(
            "Embedding model name",
            value=(
                health["openai"]["embedding_model"]
                if embedding == "openai_compatible"
                else health["embedding_model"]
            ),
            disabled=embedding == "hash",
            key=f"embedding_model_{embedding}",
        )
        if embedding == "hash":
            st.caption("The hash baseline compares words; it does not understand meaning.")
        elif embedding == "openai_compatible":
            st.caption(
                "Building this index sends document chunks to your configured API. "
                "Vector search sends the query to the same embedding endpoint."
            )
            if not health["openai"]["base_url_configured"]:
                st.info(
                    "API setup pending. Add LAB_OPENAI_BASE_URL and LAB_OPENAI_API_KEY "
                    "to .env when ready."
                )
        strategy = st.selectbox("Chunking strategy", health["chunkers"], index=2)
        size = st.slider("Chunk size (characters)", 100, 4000, 900, 50)
        overlap = st.slider("Overlap (characters)", 0, min(1500, size - 1), min(120, size - 1))
        index_type = st.selectbox(
            "Vector index",
            ["native", "exact", "hnsw"] if store in {"pgvector", "faiss"} else ["native"],
        )
        if st.button(
            "Build snapshot", type="primary", disabled=not selected, use_container_width=True
        ):
            result = api(
                "POST",
                "/indexes",
                json={
                    "name": name,
                    "document_ids": [d["id"] for d in selected],
                    "store": store,
                    "embedding": embedding,
                    "embedding_model": model,
                    "chunking": {"strategy": strategy, "size": size, "overlap": overlap},
                    "vector_index": index_type,
                },
            )
            if result:
                st.toast("Index queued. Refresh to follow its progress.")
                st.rerun()
    with inspect:
        st.subheader("Snapshots")
        if st.button("Refresh build status"):
            st.rerun()
        if not indexes:
            st.info("No indexes yet. The hash + SQLite combination works without a model.")
        for index in indexes:
            with st.expander(f"{index['status'].upper()} · {index_label(index)}", expanded=True):
                st.caption(
                    f"{index.get('chunk_count', 0)} chunks · "
                    f"{index.get('dimensions') or '—'} dimensions · "
                    f"stage: {index.get('stage', 'queued')}"
                )
                if index.get("error"):
                    st.error(index["error"])
                st.json(index["config"], expanded=False)
                if index["status"] == "ready":
                    if st.checkbox("Inspect chunks", key=f"chunks_{index['id']}"):
                        offset = st.number_input(
                            "Start at chunk",
                            0,
                            max(0, index["chunk_count"] - 1),
                            0,
                            key=f"offset_{index['id']}",
                        )
                        result = api(
                            "GET", f"/indexes/{index['id']}/chunks?offset={offset}&limit=10"
                        )
                        if result:
                            show_sources(result["chunks"])
                    if st.button("Rebuild as new snapshot", key=f"rebuild_{index['id']}"):
                        if api("POST", f"/indexes/{index['id']}/rebuild"):
                            st.rerun()
                if index["status"] in {"ready", "failed"}:
                    confirm = st.checkbox("Remove snapshot and its runs", key=f"di_{index['id']}")
                    if st.button("Delete index", key=f"dib_{index['id']}", disabled=not confirm):
                        if api("DELETE", f"/indexes/{index['id']}"):
                            st.rerun()


def compare(indexes, documents):
    title(
        "CONTROLLED EXPERIMENTS",
        "Compare retrieval",
        "One question, two pipelines. Inspect what changed and why.",
    )
    if not indexes:
        st.info(
            "Build at least one index. You can compare two retrieval methods on the same index."
        )
        return
    question = st.text_input(
        "Shared question", "Which daily limit applies now versus the archived policy?"
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Experiment A")
        a = configuration("compare_a", indexes, documents)
    with right:
        st.subheader("Experiment B")
        b = configuration("compare_b", indexes, documents)
    if st.button("Run comparison", type="primary"):
        result = api("POST", "/compare", json={"question": question, "configs": [a, b]})
        if result:
            st.session_state.comparison = result
    result = st.session_state.get("comparison")
    if result:
        if not result["same_corpus"]:
            st.warning(
                "These indexes use different document snapshots. "
                "This comparison changes the corpus."
            )
        st.download_button(
            "Export comparison JSON",
            json.dumps(result, indent=2),
            file_name=f"comparison-{result['id'][:8]}.json",
            mime="application/json",
        )
        for col, item in zip(st.columns(2, gap="large"), result["results"], strict=False):
            with col:
                if item["ok"]:
                    show_retrieval(item["result"])
                else:
                    st.error(item["error"])


def evaluations(indexes):
    title(
        "MEASURE / REVIEW / REPEAT",
        "Evaluation bench",
        "A fixed 40-question corpus. Separate development and held-out results.",
    )
    st.info(
        "Retrieval evaluation measures whether expected passages were found. It does not grade "
        "answer correctness, faithfulness, or prompt-injection resistance."
    )
    if indexes:
        selected = st.multiselect("Indexes to evaluate", indexes, format_func=index_label)
        methods = st.multiselect(
            "Methods", ["bm25", "vector", "hybrid", "mmr"], default=["bm25", "hybrid"]
        )
        split = st.selectbox("Question split", ["dev", "holdout", "all"])
        top_k = st.slider("Evaluation top k", 1, 20, 5)
        count = len(selected) * len(methods)
        st.caption(
            f"{count} configurations · maximum 8 per batch · uses default retrieval thresholds"
        )
        if st.button("Queue evaluation", type="primary", disabled=not 1 <= count <= 8):
            result = api(
                "POST",
                "/evaluations",
                json={
                    "split": split,
                    "configs": [
                        {"index_id": index["id"], "method": method, "top_k": top_k}
                        for index in selected
                        for method in methods
                    ],
                },
            )
            if result:
                st.rerun()
    if st.button("Refresh evaluations"):
        st.rerun()
    jobs = api("GET", "/evaluations") or []
    for job in jobs:
        with st.expander(
            f"{job['status'].upper()} · {job['created_at'][:19]} · {job['id'][:8]}", expanded=True
        ):
            if job["status"] in {"queued", "running"}:
                st.progress(job.get("progress", 0))
            if job.get("error"):
                st.error(job["error"])
            if job.get("report"):
                report = job["report"]
                table = [
                    {
                        "index": r["index"]["config"]["name"],
                        "method": r["config"]["method"],
                        **r["summary"],
                    }
                    for r in report["results"]
                ]
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.download_button(
                    "Full JSON report",
                    json.dumps(report, indent=2),
                    file_name=f"evaluation-{job['id'][:8]}.json",
                    key=f"export_{job['id']}",
                )
                st.download_button(
                    "Markdown summary",
                    markdown_report(report),
                    file_name=f"evaluation-{job['id'][:8]}.md",
                    key=f"md_{job['id']}",
                )
                st.json(report, expanded=False)


def history():
    title(
        "A RECORD OF YOUR EXPERIMENTS",
        "Runs & reviews",
        "Inspect local traces, resume a conversation, and record your judgment.",
    )
    conversations = api("GET", "/conversations") or []
    with st.expander("Saved conversations"):
        for conversation in conversations:
            label = conversation["turns"][0]["question"] if conversation["turns"] else "Empty"
            st.write(label)
            st.caption(f"{len(conversation['turns'])} turns · {conversation['id'][:8]}")
            if st.button("Resume conversation", key=f"resume_{conversation['id']}"):
                st.session_state.conversation_id = conversation["id"]
                st.session_state.conversation_index = conversation["index_ids"][0]
                st.session_state.resume_index = conversation["index_ids"][0]
                st.success("Conversation selected. Open Ask & inspect and choose its index.")
            st.json(conversation["turns"], expanded=False)
            confirm = st.checkbox(
                "Remove conversation and its answer runs", key=f"dc_{conversation['id']}"
            )
            if st.button(
                "Delete conversation", key=f"dcb_{conversation['id']}", disabled=not confirm
            ):
                if api("DELETE", f"/conversations/{conversation['id']}"):
                    if st.session_state.get("conversation_id") == conversation["id"]:
                        st.session_state.pop("conversation_id", None)
                        st.session_state.pop("answer", None)
                    st.rerun()
    runs = api("GET", "/runs") or []
    if not runs:
        st.info("Searches, comparisons, and completed answers will appear here.")
        return
    selected = st.selectbox(
        "Saved run",
        runs,
        format_func=lambda r: f"{r['type'].upper()} · {r['created_at'][:19]} · {r['id'][:8]}",
    )
    st.download_button(
        "Export run JSON",
        json.dumps(selected, indent=2),
        file_name=f"run-{selected['id']}.json",
        mime="application/json",
    )
    if selected["type"] == "chat":
        st.subheader(selected["question"])
        st.text(selected["answer"])
        show_sources(selected["sources"])
        with st.form("review"):
            old = selected.get("review") or {}
            choices = ["not_reviewed", "supported", "partial", "unsupported"]
            rating = st.selectbox(
                "Human support judgment",
                choices,
                index=choices.index(old.get("rating", "not_reviewed")),
            )
            notes = st.text_area("Review notes", value=old.get("notes", ""))
            if st.form_submit_button("Save review"):
                if api(
                    "POST",
                    f"/runs/{selected['id']}/review",
                    json={"rating": rating, "notes": notes},
                ):
                    st.success("Review saved.")
    st.json(selected, expanded=False)


with st.sidebar:
    st.markdown("## ◈ Knowledge Lab")
    st.caption("LOCAL DOCUMENT WORKBENCH")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Ask & inspect", "Documents", "Index studio", "Compare", "Evaluations", "Runs & reviews"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Local by design. Models stay under your control.")
    st.caption("No model downloads. Optional API inference. Synthetic demo data.")
    st.link_button("API reference", f"{API}/docs", use_container_width=True)

health = api("GET", "/health")
if not health:
    st.stop()
documents = api("GET", "/documents") or []
indexes = api("GET", "/indexes") or []
ready = [i for i in indexes if i["status"] == "ready"]
if page == "Ask & inspect":
    ask(ready, documents, health)
elif page == "Documents":
    library(documents)
elif page == "Index studio":
    studio(indexes, documents, health)
elif page == "Compare":
    compare(ready, documents)
elif page == "Evaluations":
    evaluations(ready)
else:
    history()
