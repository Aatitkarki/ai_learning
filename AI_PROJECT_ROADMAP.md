**AI application portfolio plan for Suresh Karki**

Prepared on 28 September 2026 after reviewing both pages of `sk_cv.pdf`. This is a proposed learning and project roadmap; the applications and measurements described below have not been built or verified yet.

The goal is to build three useful applications that you can demonstrate, explain, change, and debug in an interview. Your banking, remittance, Flutter, and software delivery experience gives the projects a consistent domain. The AI concepts will be introduced in small steps.

Planning assumptions: target Python / AI application engineering roles; spend approximately 8-10 hours per week; begin with local demos and synthetic data. Your Python confidence, interview deadline, hardware, and preference for paid APIs may change the estimates. Model names and dependency versions will be selected when implementation starts.

Your current CV presents broad AI expertise, including specific contributions at previous employers. New projects establish current skills. They do not establish what happened at those employers. Keep employment bullets specific to your actual contribution, and describe these new applications as personal projects once completed. Strong mobile engineering experience is a useful foundation for this transition.

We will build in this order:

| Order | Application | Purpose | Main concepts | Estimated learning and build effort |
| --- | --- | --- | --- | --- |
| 1 | Support Transcript Summarizer | Turn a support conversation into a reliable handover | Model requests, prompts, structured output, validation | 20-30 hours |
| 2 | Local Document Knowledge Assistant | Answer questions from a small document collection, with evidence | RAG, embeddings, retrieval, local inference, streaming, evaluation | 35-50 hours |
| 3 | Remittance Support Assistant | Combine document answers with mock transaction tools | Tool calling, workflow state, LangGraph, failure handling | 30-45 hours |
| Final pass | Portfolio preparation | Make the three apps easy to run and discuss | Reproducibility, demos, measured results, interview practice | 15-20 hours |

Total planning estimate: 100-145 focused hours, approximately 10-18 weeks at 8-10 hours per week. This includes learning and explanation practice, not just code generation. Re-estimate after the first working feature; unfamiliar Python or limited local hardware can extend the schedule. If time becomes constrained, finish apps 1 and 2 before adding app 3.

**1. Support Transcript Summarizer**

A support employee pastes a fictional customer conversation and gets a concise handover: customer issue, key facts, actions already taken, unresolved questions, and next steps. Evidence links point back to transcript lines. This is the smallest application and provides the foundation for the other two.

For example, a customer says a transfer has not arrived. The summary should preserve the stated amount and reference, identify what the agent already checked, and leave the reason for the delay unknown when the conversation does not explain it. Suggested next steps must be visibly distinct from actions that already happened.

The first version will include:

- A text input or `.txt` upload and a result screen.
- A Python/FastAPI endpoint and Pydantic request/response models.
- One direct model integration, chosen after checking local hardware and API budget.
- Structured output with explicit missing values, plus validation before display.
- Transcript line references for factual claims and editable summary output.
- Clear behavior for empty input, oversized input, invalid model output, and an unavailable model.
- A small local history using SQLite, with deletion.

Start with one short transcript and one model call. Add the API and interface after the output is understandable. Use local Ollama if it runs acceptably on the machine; use an OpenAI API option if available and within the chosen budget. The OpenAI-specific CV claim only gains project evidence if that integration is actually implemented and exercised. A Flutter interface can use your existing strengths, but its first version only needs input, output, and history screens.

Keep the initial input within a defined size limit. Audio transcription, arbitrary PDF parsing, long-document summarization, and automatic sending of follow-up messages are later extensions. A summarizer does not need a vector database or agent framework for this first use case.

Prepare 20 synthetic transcripts with hand-written expected facts, including missing amounts, contradictory statements, complaints, and ambiguous dates. Check schema validity, preserved names and numbers, unsupported claims, and evidence references. Measure actual latency and token usage where available. Valid JSON alone does not demonstrate a factually correct summary.

The interview demo is a three-minute flow: submit a conversation, inspect the structured handover and supporting lines, then show an incomplete conversation where the app leaves information unknown.

Before moving on, you should be able to explain a model request, tokens, context limits, prompts, structured output, and why validation cannot guarantee factual accuracy. You should be able to add one summary field and diagnose one failed response with limited assistance.

CV coverage: Python, FastAPI, Pydantic, LLM service integration, summarization, REST APIs, and eventually Docker. OpenAI integration is conditional on implementing that provider.

**2. Local Document Knowledge Assistant**

This is the flagship project and directly develops the personal document assistant described in your CV. A user uploads fictional banking/remittance policies and asks questions. Answers link to the document, page, and relevant passage. Unsupported questions receive an explicit statement that the available documents do not establish an answer.

RAG means retrieving useful passages and giving them to the language model as evidence for an answer. We will implement and inspect those steps individually before introducing a framework.

The first version will include:

- Upload and ingestion of text-based PDFs and Markdown files, with visible ingestion status.
- Extracted text divided into chunks that preserve document and page metadata.
- Embeddings for documents and questions, stored in PostgreSQL with pgvector.
- Search results visible beside the generated answer so retrieval can be inspected.
- Answers with source references and links to supporting passages.
- Local generation and local embeddings through Ollama, subject to hardware suitability.
- Streamed answer text and basic conversation follow-ups.
- Document deletion/re-indexing and a way to reproduce an evaluation run.

Ollama documents embedding generation for semantic search and RAG; pgvector provides vector similarity search in PostgreSQL. These are the proposed local embedding and storage components. See the [Ollama embedding documentation](https://docs.ollama.com/capabilities/embeddings) and [pgvector documentation](https://github.com/pgvector/pgvector).

Use one main vector store. First inspect a tiny embedding/search example in plain Python, then build the working retrieval pipeline. Introduce selected LangChain components after the flow is understood. LangChain is the primary framework for this app; a LlamaIndex comparison is an optional experiment, not a second full implementation.

Build the application in four milestones:

1. Extract one document, inspect its chunks, and retrieve relevant passages without generating answers.
2. Generate a cited answer from those passages and inspect unsupported or incorrect outputs.
3. Add multiple documents, streaming, follow-up questions, and a usable interface.
4. Compare retrieval settings on a fixed evaluation set and produce a results report.

Start with approximately 10-15 fictional documents. Explicitly label their rules and amounts as demonstration data. Scanned documents requiring OCR, complex table extraction, and a shared multi-user deployment are outside the initial scope. A hosted model may help development if local generation is too slow, but the app only counts as demonstrating local inference after its local path has been exercised. Keep tracing and embeddings local when demonstrating an entirely local path.

Prepare 40 questions with expected supporting passages and expected behavior: 25 answerable questions, 10 unsupported questions, and 5 cases involving conflicting document versions or instructions embedded in a document. Keep a portion separate from prompt/chunk tuning and report that portion separately.

Record whether the expected passage appears among the retrieved results, whether each answer's factual claims have support, whether citations point to appropriate evidence, and how the app handles unsupported questions. Record time to first token and total response time. Compare at least two chunking or retrieval settings while keeping the model and question set fixed.

Add Ragas for selected evaluation metrics after the hand-reviewed baseline exists. Its metric catalogue includes context precision, context recall, response relevancy, and faithfulness. Treat automated scores as evidence to inspect alongside examples, not as proof of correctness. See the [Ragas metric documentation](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/).

Add tracing so you can inspect the retrieval and generation steps. A LangSmith integration on synthetic examples can support the observability claim on your CV; its documentation describes request traces and their component steps. Local logs remain useful for offline operation. See [LangSmith observability](https://docs.langchain.com/langsmith/observability).

The interview demo is a five-minute flow: ask a question, inspect the retrieved passage and answer, open the cited page, ask an unsupported question, and explain one failure found during evaluation. Show the measured comparison between two retrieval settings.

Before moving on, you should be able to draw the ingestion and question-answering flows, explain embeddings and chunking, distinguish retrieval failures from generation failures, and change a retrieval setting yourself. You should also be able to explain why having a citation does not automatically make an answer correct.

CV coverage: RAG, document retrieval, embeddings, vector search, pgvector, LangChain, Ollama, local inference, streaming, and evaluation/observability once implemented. The LlamaIndex and llama.cpp claims remain conditional on the later comparison exercises.

**3. Remittance Support Assistant**

A fictional remittance service lets a user ask about a transfer, request a sample fee quote, read policy information, or prepare a support ticket. This connects your existing remittance and API experience with AI workflow development.

The backend exposes four small tools:

| Tool | Source of truth | Expected behavior |
| --- | --- | --- |
| `get_transfer_status` | Seeded mock transfer records | Return a known status or an explicit not-found result |
| `calculate_fee_quote` | Fictional fee/rate table and ordinary Python calculations | Calculate reproducibly, with labeled demonstration rates |
| `search_policy` | Retrieval service developed in app 2 | Return supporting policy passages |
| `create_support_ticket` | Local application database | Show a draft and require user confirmation before creation |

Use a small fixed demo identity and transfer ownership mapping, enforced in backend code. Tool arguments come from the model, but validation, record access, calculations, and permitted actions belong to the application. The demo uses fake customers and transfers and does not initiate money movement.

The model first proposes a tool and its arguments. Application code validates and executes the allowed request. The result is then used to produce a response. Build and inspect one such cycle in plain Python before introducing LangGraph.

Use LangGraph when adding workflow state, missing-information questions, bounded tool loops, and the support-ticket confirmation step. Its documentation distinguishes predetermined workflows from agents whose tool use is dynamic, and describes persistence and streaming capabilities. We will use a small controlled workflow for this app. See [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents).

The first version will include a Flutter chat interface, visible tool activity, validated inputs, saved conversation state, tool timeouts, a maximum step count, and clear failure messages. Persist the ticket draft during confirmation; repeated confirmation or retried requests must not create duplicate tickets.

Reuse the retrieval capability from app 2 after it works; preserve a self-contained startup guide and seeded data for this demo. Introduce reusable code only where an actual second use appears.

Prepare 25 scripted scenarios covering successful tool use, a missing transfer reference, unknown transfer, another demo user's transfer, invalid quote amount, unavailable backend, irrelevant request, tool-loop limit, cancelled ticket creation, and repeated confirmation. Record correct tool/argument selection and successful completion separately from response wording. Record whether access and confirmation rules hold.

The interview demo is a five-minute flow: track a fictional transfer, ask a related policy question, prepare and confirm a support ticket, then show an unavailable tool or invalid reference. Explain the workflow from the execution trace. A fee quote should show numbers produced by code rather than invented in a response.

Before considering this complete, you should be able to explain why a tool was needed, how the model's request differs from execution, how state is retained, how access and confirmation are enforced, and what stops repeated calls. You should be able to add one small mock tool and demonstrate a failure case.

CV coverage: tool calling, LangGraph, chatbot APIs, application integration, streaming, clean architecture, and Flutter integration. Evaluation and tracing carry forward from the earlier apps.

The three projects cover the CV selectively:

| CV claim | Planned evidence | How to describe it before that evidence exists |
| --- | --- | --- |
| Python, FastAPI, Pydantic | APIs and validation across the three apps | Match your actual current ability |
| Summarization and OpenAI API | App 1, with OpenAI only if its adapter is used | Provider integration remains planned until exercised |
| RAG, embeddings, vector search | App 2, reused in app 3 | Learning or prototyping until you can explain the pipeline |
| LangChain | Selected components in app 2 | Do not imply advanced framework experience from one tutorial |
| LangGraph and tool calling | App 3 | Planned until the workflow is implemented and understood |
| pgvector | App 2's primary vector store | Hands-on project experience after completion |
| Ollama and local inference | App 2's verified local path | Limited to the runtimes and models actually used |
| Ragas and LangSmith | App 2 evaluation and trace analysis | Planned until you run and interpret them |
| LlamaIndex, llama.cpp | Optional controlled comparison exercises | Familiarity only after completing and explaining the exercise |
| FAISS, Chroma, Pinecone | Optional retrieval comparison | Defer or remove from the skills list if unused |
| Hugging Face, PyTorch | Separate learning work if relevant to target roles | An API-based or Ollama app does not demonstrate PyTorch skill |
| Docker, Git, reproducibility | Runnable projects, meaningful history, documented setup | Claim only the practices actually used |

After the core apps, choose an optional comparison only when it answers a question you can measure. For example, replace part of app 2 with LlamaIndex while keeping the documents and questions fixed, compare pgvector with one alternative, or compare Ollama and llama.cpp using comparable model/settings on the same machine. Record the setup and limitations; results from one machine are not a universal benchmark. Pinecone may introduce an account and cost, so it is optional.

PyTorch model training is a different learning track. If your target becomes ML engineering, revise this roadmap to include a real training/evaluation project and the necessary mathematical foundations. Simply downloading a model does not establish training experience.

For every feature, we will follow a repeatable learning cycle: explain the concept with a small example, implement the smallest working version, inspect intermediate data, exercise a failure, and have you explain or modify the feature yourself. I can help build and debug the applications, but your ability to defend a design choice and make a small change is part of the completion criteria.

Each completed app should include:

- Reproducible setup, pinned dependencies, an example environment file without secrets, and seeded synthetic data.
- A README with the problem, architecture, limitations, and exact startup instructions.
- An architecture diagram and a small API example.
- Relevant automated checks plus a repeatable evaluation set with expected outcomes.
- A dated results report containing actual measurements and failure examples.
- A short demo recording as backup for a live demonstration.
- Interview notes explaining one design tradeoff, one bug, one improvement, and your personal contribution.

Run local models on the host when that is the practical hardware path, and document how the containerized backend reaches them. A public deployment is optional and can be planned after the local demo is stable. The first implementation milestone is app 1's single-transcript-to-validated-summary flow, followed by its evidence check and explanation exercise.

**Implementation update — 28 September 2026**

Project 2 now has a local experiment workbench in
[local-document-knowledge-assistant](local-document-knowledge-assistant/README.md).
At Suresh's request, its scope expands the original single-store recommendation to interchangeable
pgvector, Chroma, Qdrant, FAISS, and a SQLite/NumPy reference backend, with multiple chunking and
retrieval methods, side-by-side comparisons, local traces, and a prepared 40-question evaluation set.
The default path uses deterministic hash vectors and quoted evidence without a language model.
Model downloads, live local inference, optional database integration tests, and quality benchmarks
remain deferred. See the app's [verification record](local-document-knowledge-assistant/reports/VERIFICATION.md)
for the distinction between implemented features, automated core checks, and pending experiments.

The app also has an OpenAI-compatible chat/embedding API adapter with configurable base URL,
key, and independent model names in `.env`. These values remain blank until provided; startup
does not contact a provider. API generation and API indexing are explicit choices in the UI.
