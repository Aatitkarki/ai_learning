# Understand and explain the first project

Read these files in order: `models.py`, `prompts.py`, `provider.py`, `api.py`,
`repository.py`, then `ui.py`. Start with `scripts/summarize.py` to see one call in isolation.

## Core concepts

**Model request:** an HTTP request containing a model name, instructions, transcript,
generation settings and output schema. Ollama runs inference locally and returns text
plus execution metadata. `provider.py` is the only production model integration.

**Tokens:** the model's units of text, not necessarily whole words or characters.
Token counts come from Ollama's `prompt_eval_count` and `eval_count`. The app does not
pretend a character count is an exact token count.

**Context window:** a bound on how much instruction, input and output a model can handle
in one request. The app deliberately limits transcript size and configures context and
output budgets. It does not split, retrieve, or summarize arbitrary long documents.

**Prompt:** instructions describing what counts as a fact, when to use null, how to cite
evidence, and why future work must not appear among completed actions. The transcript is
JSON-encoded data, but that alone cannot guarantee immunity from embedded instructions.

**Structured output:** the model is asked to return JSON with a known shape. Pydantic
validates it again before display. A response can satisfy the schema and still be wrong.

**Validation:** deterministic checks reject malformed structure, absent citations, values
not copied from cited lines, and unsupported numeric tokens. They cannot establish the
meaning of arbitrary natural-language claims. `test_validation_cannot_guarantee_semantics`
is an intentionally passing test documenting that limitation.

## Explain a tradeoff

The first UI uses Streamlit rather than adding a second language and framework. That
makes the project approachable for learning Python and keeps the backend independent.
The cost is less control over interaction and visual layout than a custom Flutter client.

Exact extracted values make names and references easier to audit, but they can reject a
semantically equivalent paraphrase or a harmless currency reformatting. This favors
preservation for financial-support handovers. Rejecting output does not fix the model;
the failure remains visible to the user.

## Diagnose a failure

If the app returns `invalid_model_output`, distinguish malformed JSON, a missing field,
truncation, and unsupported evidence. Reproduce it with the synthetic evaluation runner;
its diagnostic and case ID let you inspect the issue without logging real conversations.
Check the original lines before changing the validation rules. Never weaken a citation
check simply to make an evaluation pass.

History edits use a revision number. Without it, two tabs could overwrite one another.
The API now returns 409 for a stale edit, and an automated test exercises that case.

An actual implementation bug: the interface used a plain check mark as a Streamlit alert
icon. That character was not accepted as an emoji, so the ready-model screen crashed even
though the unavailable-model screen worked. The UI test reproduced that state, and a
supported icon fixed it. Explain why testing only the unavailable state would have missed it.

## One improvement to discuss

Add a held-out, independently reviewed evaluation set and compare models while keeping
the prompt and data fixed. This would provide stronger evidence than tweaking prompts
against the same development cases. Human review should test support and attribution,
not just word overlap.

## Explain your contribution honestly

This implementation was created with AI coding assistance. Before claiming personal
proficiency, run it, inspect an actual failed case, and make a small change yourself.
Describe what you understand, changed and verified; this project does not prove historical
contributions at an employer or experience with providers not implemented here.

## Small extension exercise

Add a `preferred_contact_channel` sourced field:

1. Define it as `SourcedValue` in `Handover`.
2. Ask the model to copy an explicit channel or return null.
3. Include it in exact-value validation, rendering, and the result interface.
4. Add one synthetic example with a stated channel and one without one.
5. Extend the request/response fixture and live expectations, then rerun checks.
6. Explain why a guessed channel should not be presented as known.

This schema change also affects saved history. For a disposable demo, create a fresh
database via `DATABASE_PATH`; for retained data, add an explicit version/migration plan.
Do not silently delete a user's history to accommodate a schema change.
