import re
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_TRANSCRIPT_CHARS = 6_000
MAX_TRANSCRIPT_BYTES = 8_000
MAX_TRANSCRIPT_LINES = 80
MAX_UPLOAD_BYTES = 24_000
LineNumber = Annotated[int, Field(strict=True, ge=1, le=MAX_TRANSCRIPT_LINES)]
ShortText = Annotated[str, Field(min_length=1, max_length=500)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SummaryRequest(StrictModel):
    transcript: str = Field(min_length=1, max_length=MAX_TRANSCRIPT_CHARS)
    title: str = Field(default="Support conversation", min_length=1, max_length=100)

    @field_validator("transcript", mode="before")
    @classmethod
    def normalize_transcript(cls, value):
        if not isinstance(value, str):
            return value
        # Bound the raw input before normalizing; preserve internal blank lines.
        if len(value) > MAX_TRANSCRIPT_CHARS:
            raise ValueError(f"Transcript exceeds {MAX_TRANSCRIPT_CHARS:,} characters.")
        value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(value.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
            raise ValueError(f"Transcript exceeds {MAX_TRANSCRIPT_BYTES:,} UTF-8 bytes.")
        if "\x00" in value:
            raise ValueError("Transcript must be plain text without NUL characters.")
        if len(value.split("\n")) > MAX_TRANSCRIPT_LINES:
            raise ValueError(f"Transcript exceeds {MAX_TRANSCRIPT_LINES} lines.")
        return value


class SourcedValue(StrictModel):
    value: ShortText | None
    source_lines: list[LineNumber] = Field(max_length=8)

    @model_validator(mode="after")
    def require_evidence(self):
        if self.value is None and self.source_lines:
            raise ValueError("Unknown values must have no evidence lines.")
        if self.value is not None and not self.source_lines:
            raise ValueError("Known values require evidence lines.")
        return self


class Claim(StrictModel):
    text: ShortText
    source_lines: list[LineNumber] = Field(min_length=1, max_length=8)


class Handover(StrictModel):
    customer_issue: Claim | None
    customer_name: SourcedValue
    amount: SourcedValue
    transfer_reference: SourcedValue
    stated_date: SourcedValue
    delay_reason: SourcedValue
    key_facts: list[Claim] = Field(max_length=8)
    actions_taken: list[Claim] = Field(max_length=8)
    unresolved_questions: list[ShortText] = Field(max_length=6)
    contradictions: list[Claim] = Field(max_length=6)
    suggested_next_steps: list[ShortText] = Field(max_length=6)


class Metrics(StrictModel):
    provider: str = "ollama"
    model: str
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    prompt_version: str


class SavedSummary(StrictModel):
    id: str
    title: str
    transcript: str
    summary: Handover
    metrics: Metrics
    draft_markdown: str
    is_edited: bool
    revision: int
    created_at: datetime
    updated_at: datetime


class HistoryItem(StrictModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_edited: bool


class DraftUpdate(StrictModel):
    draft_markdown: str = Field(min_length=1, max_length=20_000)
    revision: int = Field(ge=1, le=2_147_483_647)


class EvidenceError(ValueError):
    """The schema is valid but a mechanical evidence check failed."""


def validate_evidence(summary: Handover, transcript: str) -> None:
    """Check bounds, exact extracted values and numeric support, not semantic truth."""
    lines = transcript.split("\n")

    def sources(refs: list[int]) -> str:
        if any(ref > len(lines) or not lines[ref - 1].strip() for ref in refs):
            raise EvidenceError("A citation refers to an absent or empty transcript line.")
        if len(refs) != len(set(refs)):
            raise EvidenceError("Duplicate evidence references are not allowed.")
        return "\n".join(lines[ref - 1] for ref in refs)

    for field in ("customer_name", "amount", "transfer_reference", "stated_date", "delay_reason"):
        fact = getattr(summary, field)
        sources(fact.source_lines)
        if fact.value is not None:
            # Match a single quoted span, never a synthetic combination across lines.
            if not any(fact.value in lines[ref - 1] for ref in fact.source_lines):
                raise EvidenceError(f"{field} is not an exact quote from its evidence.")
    claims = summary.key_facts + summary.actions_taken + summary.contradictions
    if summary.customer_issue:
        claims = [summary.customer_issue, *claims]
    for claim in claims:
        evidence = sources(claim.source_lines)
        tokens = re.findall(r"(?<!\w)[+-]?\b[\w.,:/-]*\d[\w.,:/-]*\b", claim.text)
        for token in tokens:
            if not re.search(r"(?<![\w+-])" + re.escape(token) + r"(?!\w)", evidence, re.I):
                raise EvidenceError(f"A number or reference ({token}) is absent from cited lines.")
