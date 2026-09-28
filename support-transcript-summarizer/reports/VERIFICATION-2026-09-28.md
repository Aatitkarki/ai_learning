# Verification — 28 September 2026

Environment: macOS 26.0, Apple silicon (arm64), 16 GiB memory, Python 3.11.15, uv 0.11.8.
Dependencies are resolved in `uv.lock`; the final environment passes `uv sync --frozen --offline`.

## Executed checks

| Check | Observed result |
| --- | --- |
| `pytest -q` | 49 passed in 0.54 seconds |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 27 files already formatted |
| Documented `scripts/dev.py` launcher | API and Streamlit started successfully |
| Streamlit `/_stcore/health` | HTTP 200, `ok` |
| Interface script against the actual running API | Loaded without exceptions; real unavailable-model status and Create action visible |
| FastAPI `/health` with Ollama absent | HTTP 200; `model_status.available=false` |
| Actual POST with Ollama absent | HTTP 503, `model_unavailable` |
| Ollama runtime `/api/version` | `0.6.8`, started on loopback using the project-local model directory |
| Actual POST with Ollama running but the model absent | HTTP 503, `model_missing` |
| History after that failed POST | Empty list; no successful record saved |
| Docker execution | Not run: no working Docker daemon or Compose command in this environment |

The automated suite uses explicit provider test doubles and mocked transport responses.
It verifies schema/evidence checks, numeric signs, limits, malformed responses, timeouts,
uploads/encoding, persistence, pagination, editing conflicts, deletion, export, the dataset,
and the Streamlit widget workflow. The tests do not establish model factual accuracy.

The Streamlit test runner exercised the actual interface script and API routes with a
test database. Pixel-level browser inspection was not performed. An upstream Starlette
deprecation warning recommends `httpx2` for future test-client compatibility; it did not
prevent the pinned suite from passing.

## Actual failure found and fixed

The first complete UI run failed in the model-ready state because `st.success(icon="✓")`
does not accept that character as an emoji. The unavailable-model state had rendered
normally. The ready-state widget test exposed the crash; changing the icon to a supported
emoji fixed it. The full 49-test suite then passed.

## Model evaluation status

Two real 20-case evaluations ran with `qwen2.5:0.5b`:

| Prompt | Schema valid | Evidence checks passed | All expected checks passed |
| --- | --- | --- | --- |
| [1.0.0 baseline](baseline/evaluation-20260928T083758Z.md) | 17/20 | 2/20 | 0/20 |
| [1.1.0 with worked examples](few-shot/evaluation-20260928T084936Z.md) | 19/20 | 10/20 | 0/20 |

The current prompt improved mechanical validation, but this small model still missed
facts, invented completed actions, and mishandled conflicts. It is not a reliable demo
model. These are development checks, not factual-accuracy measurements; full human
review remains pending. The two runs also used different dataset revisions, with added
checks for invented actions in the later run. Each JSON report records its dataset hash.

The initial incomplete-conversation smoke test fabricated an amount and date; the
validator rejected it. Its raw response is preserved in `initial-smoke-response.json`.
The later run's median latency among the ten mechanically valid outputs was 1,106 ms,
with 8,975 input and 1,736 output tokens recorded for those outputs only. Failed outputs
are excluded from these token totals. See the linked reports for per-case results.

The user requested no model downloads and deferred model setup. The 0.5b download had
already completed when that instruction arrived; no further downloads were performed.
Project development and Ollama processes were stopped. Existing local weights remain
in the ignored `.runtime/models` directory. No successful demo recording was produced;
the recording script and three-minute demo instructions are included for later use.

The verification runtime is the smaller official Ollama 0.6.8 macOS archive, used after
repeated interruptions of the current runtime download. The normal setup guide points to
the current Ollama installer. The live baseline is configured as `qwen2.5:0.5b`; the
larger `qwen2.5:3b` default is not downloaded or exercised by this verification run.
