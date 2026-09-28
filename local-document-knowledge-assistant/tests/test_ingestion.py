import pytest
from pydantic import ValidationError

from knowledge_lab.config import LabError
from knowledge_lab.ingestion import chunk_document, extract
from knowledge_lab.models import ChunkConfig


@pytest.mark.parametrize("strategy", ["fixed", "sentence", "markdown"])
def test_chunk_offsets_pages_overlap_and_coverage(strategy):
    first = "# First\n\n" + "A complete test sentence with a fee of five credits. " * 20
    second = "# Second\n\n" + "Another policy page preserves its independent page number. " * 10
    doc = {
        "id": "doc",
        "filename": "test.pdf",
        "pages": [{"number": 1, "text": first}, {"number": 3, "text": second}],
    }
    config = ChunkConfig(strategy=strategy, size=200, overlap=40)
    chunks = chunk_document(doc, "index", config)
    assert {c.page for c in chunks} == {1, 3}
    assert len({c.id for c in chunks}) == len(chunks)
    for page in doc["pages"]:
        covered = set()
        for chunk in [c for c in chunks if c.page == page["number"]]:
            assert chunk.text == page["text"][chunk.start : chunk.end]
            assert 0 < len(chunk.text) <= config.size
            covered.update(range(chunk.start, chunk.end))
        assert all(i in covered for i, char in enumerate(page["text"]) if not char.isspace())


def test_markdown_does_not_cross_sections():
    document = {
        "id": "doc",
        "filename": "policy.md",
        "pages": [{"number": 1, "text": "# One\n\nFirst policy.\n\n## Two\n\nSecond policy."}],
    }
    chunks = chunk_document(document, "index", ChunkConfig())
    assert len(chunks) == 2
    assert [c.heading for c in chunks] == ["One", "Two"]


@pytest.mark.parametrize(
    "filename,content",
    [
        ("empty.txt", b""),
        ("bad.exe", b"words"),
        ("binary.txt", b"a\x00b"),
        ("bad.txt", b"\xff"),
        ("bad.pdf", b"not a PDF"),
        ("blank.md", b" \n "),
    ],
)
def test_reject_bad_documents(settings, filename, content):
    with pytest.raises(LabError):
        extract(filename, content, settings)


def test_pdf_missing_dependency_has_actionable_error(settings, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(LabError, match="'pdf' extra"):
        extract("sample.pdf", b"%PDF-1.4\n", settings)


def test_utf8_normalization_and_size_limit(settings):
    assert extract("a.md", b"\xef\xbb\xbfhello\r\nworld", settings)[0].text == "hello\nworld"
    settings.max_upload_bytes = 5
    with pytest.raises(LabError):
        extract("a.txt", b"123456", settings)


def test_invalid_overlap():
    with pytest.raises(ValidationError):
        ChunkConfig(size=100, overlap=100)
