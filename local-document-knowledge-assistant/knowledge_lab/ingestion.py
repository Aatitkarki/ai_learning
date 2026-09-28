import io
import re
import uuid
from pathlib import Path

from .config import LabError, Settings
from .models import Chunk, ChunkConfig, Page


def extract(filename: str, content: bytes, settings: Settings) -> list[Page]:
    if not content or len(content) > settings.max_upload_bytes:
        raise LabError("Upload a nonempty file of at most 10 MB.")
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".txt", ".markdown"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise LabError("Text files must use UTF-8 encoding.") from exc
        if "\x00" in text:
            raise LabError("Binary content is not supported in text files.")
        pages = [Page(number=1, text=text.replace("\r\n", "\n").replace("\r", "\n"))]
    elif suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise LabError("This file does not have a PDF header.")
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise LabError(
                "PDF support is not installed. Install the 'pdf' extra when ready.", 503
            ) from exc
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise LabError("Password-protected PDFs are not supported.")
            if len(reader.pages) > 300:
                raise LabError("PDFs must contain at most 300 pages.")
            pages, count = [], 0
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                count += len(text)
                if count > settings.max_text_chars:
                    raise LabError("Extracted document text exceeds the 2 million character limit.")
                pages.append(Page(number=number, text=text))
        except LabError:
            raise
        except Exception as exc:
            raise LabError("Unable to read this PDF. Use a valid, text-based PDF.") from exc
    else:
        raise LabError("Supported file types: .pdf, .md, .markdown and .txt.")
    if sum(len(p.text) for p in pages) > settings.max_text_chars:
        raise LabError("Extracted document text exceeds the 2 million character limit.")
    if not any(p.text.strip() for p in pages):
        raise LabError("No text found. Scanned PDFs need OCR, which is outside this app's scope.")
    return pages


def _windows(text: str, start: int, end: int, config: ChunkConfig):
    """Split by offsets, preserving exact source substrings and bounded overlap."""
    while start < end:
        stop = min(start + config.size, end)
        if stop < end and config.strategy in {"sentence", "markdown"}:
            region = text[start:stop]
            boundaries = [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n\n", region)]
            candidates = [b for b in boundaries if b >= max(config.size // 2, config.overlap + 1)]
            if candidates:
                stop = start + candidates[-1]
        left, right = start, stop
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            yield left, right
        if stop >= end:
            break
        start = max(start + 1, stop - config.overlap)


def chunk_document(document: dict, index_id: str, config: ChunkConfig) -> list[Chunk]:
    result = []
    for page in document["pages"]:
        text = page["text"]
        headings = list(re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE))
        if config.strategy == "langchain":
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError as exc:
                raise LabError(
                    "Install the 'langchain' extra to use its recursive splitter.", 503
                ) from exc
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.size, chunk_overlap=config.overlap, add_start_index=True
            )
            spans = [
                (d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content))
                for d in splitter.create_documents([text])
            ]
        else:
            cuts = [0]
            if config.strategy == "markdown":
                cuts += [m.start() for m in headings if m.start() > 0]
            cuts += [len(text)]
            spans = [
                span
                for a, b in zip(cuts, cuts[1:], strict=False)
                for span in _windows(text, a, b, config)
            ]
        for start, end in spans:
            if start < 0:
                raise LabError("Splitter could not preserve source offsets.")
            heading = next((m.group(1) for m in reversed(headings) if m.start() <= start), "")
            ordinal = len(result)
            key = f"{index_id}:{document['id']}:{page['number']}:{start}:{end}"
            result.append(
                Chunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                    document_id=document["id"],
                    filename=document["filename"],
                    page=page["number"],
                    ordinal=ordinal,
                    start=start,
                    end=end,
                    text=text[start:end],
                    heading=heading,
                )
            )
    return result
