from concurrent.futures import ThreadPoolExecutor

from knowledge_lab.models import ChatRequest


def test_stream_can_resume_on_a_different_thread(indexed):
    lab, index, _ = indexed
    stream = lab.chat(ChatRequest(question="refund", config={"index_id": index["id"]}))
    assert next(stream)["type"] == "retrieval"
    with ThreadPoolExecutor(max_workers=1) as pool:
        remaining = pool.submit(list, stream).result(timeout=5)
    assert remaining[-1]["type"] == "complete"


def test_deleting_index_during_stream_does_not_resurrect_evidence(indexed):
    lab, index, _ = indexed
    stream = lab.chat(ChatRequest(question="refund", config={"index_id": index["id"]}))
    assert next(stream)["type"] == "retrieval"
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lab.delete_index, index["id"]).result(timeout=5)
    assert list(stream)[-1]["type"] == "error"
    assert not lab.repo.list("runs") and not lab.repo.list("conversations")


def test_parallel_turns_do_not_overwrite_conversation_history(indexed):
    lab, index, _ = indexed
    request = ChatRequest(question="refund", config={"index_id": index["id"]})
    conversation_id = list(lab.chat(request))[-1]["result"]["conversation_id"]
    followup = request.model_copy(
        update={"conversation_id": conversation_id, "question": "How long does it take?"}
    )
    first, second = lab.chat(followup), lab.chat(followup)
    next(first)
    next(second)
    assert list(first)[-1]["type"] == "complete"
    assert list(second)[-1]["type"] == "error"
    assert len(lab.require("conversations", conversation_id)["turns"]) == 2
