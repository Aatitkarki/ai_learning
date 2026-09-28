from summarizer.models import Claim, Handover


def cited(claim: Claim) -> str:
    return f"{claim.text} [" + ", ".join(f"L{line}" for line in claim.source_lines) + "]"


def to_markdown(summary: Handover) -> str:
    parts = [
        "# Support handover",
        "",
        "## Customer issue",
        cited(summary.customer_issue) if summary.customer_issue else "Unknown / not stated.",
    ]
    parts.extend(["", "## Key details"])
    for label, field in [
        ("Customer", "customer_name"),
        ("Amount", "amount"),
        ("Reference", "transfer_reference"),
        ("Date as stated", "stated_date"),
        ("Delay reason", "delay_reason"),
    ]:
        fact = getattr(summary, field)
        value = fact.value if fact.value is not None else "Unknown / not established"
        refs = (
            " [" + ", ".join(f"L{line}" for line in fact.source_lines) + "]"
            if fact.source_lines
            else ""
        )
        parts.append(f"- {label}: {value}{refs}")
    for label, items in [
        ("Other key facts", summary.key_facts),
        ("Actions already taken", summary.actions_taken),
        ("Conflicting statements", summary.contradictions),
    ]:
        parts.extend(["", f"## {label}"])
        parts.extend([f"- {cited(item)}" for item in items] or ["None stated."])
    for label, items in [
        ("Unresolved questions", summary.unresolved_questions),
        ("Suggested next steps — not completed", summary.suggested_next_steps),
    ]:
        parts.extend(["", f"## {label}"])
        parts.extend([f"- {item}" for item in items] or ["None suggested."])
    return "\n".join(parts)


def export_markdown(record) -> str:
    label = (
        "Human-edited draft; edits have not been automatically evidence-checked."
        if record.is_edited
        else "Model-generated handover; review against the source."
    )
    transcript = "\n".join(
        f"L{i}: {line}" for i, line in enumerate(record.transcript.split("\n"), 1)
    )
    return (
        f"{label}\n\n{record.draft_markdown}\n\n## Source transcript\n\n"
        + "\n".join(f"    {line}" for line in transcript.split("\n"))
        + "\n"
    )
