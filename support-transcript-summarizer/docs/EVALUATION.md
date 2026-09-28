# Review the handover, not just the JSON

`evaluation/cases.json` contains 20 synthetic cases and expectations authored before the
first model run. The baseline covers missing values, contradictory amounts/statuses,
complaints, ambiguous dates, corrections, proposed/completed actions, leading zeroes,
decimal punctuation, Unicode names, speculation, multiple transfers, non-transfer support,
blank lines, and an instruction embedded in the conversation.

The runner separately records schema validity, mechanical evidence validity, expected
scalar-value matches, selected fact/action keyword matches, latency and available tokens.
An invalid response is a failed case. The keyword checks may reject an acceptable paraphrase
or miss an unsupported claim. Missing token counts are not replaced with estimated usage.

For each output, compare it with the original case transcript and complete this rubric:

| Dimension | Human review question |
| --- | --- |
| Coverage | Did it preserve the issue and all material handover facts? |
| Names and numbers | Are names, references, amounts, punctuation and leading zeroes preserved? |
| Attribution | Does the output distinguish customer claims, agent observations and speculation? |
| Evidence | Does each cited line actually support the associated claim, including negation? |
| Action status | Are completed actions separate from intentions, requests and promises? |
| Unknowns | Did absent or ambiguous details remain unknown? |
| Conflicts | Are both sides of a disagreement visible without silently resolving it? |
| Unsupported claims | Did the model add a cause, promise, completion, policy or date? |
| Proposed steps | Are suggestions plausible, clearly proposed, and free of invented policy? |

Record a pass/fail for each dimension, the exact unsupported claim if any, its cited lines,
and a short explanation in a separate dated review file. Human review should not overwrite
the raw generated JSON report. Do not treat a citation's existence as proof of support.

When changing a prompt or model, keep the data fixed and produce a new dated report.
Record cold versus warm generation when comparing latency. A one-machine, 20-case run
does not establish universal performance. A future improvement is a held-out test set
reviewed by another person; this first version has no such independent assessment.
