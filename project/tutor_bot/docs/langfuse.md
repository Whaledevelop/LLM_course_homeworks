# Langfuse observability

Tutor Bot always writes observability events to
`data/history/observability_events.jsonl`. Langfuse is an optional second sink.
If Langfuse is unavailable, application requests and local logging continue.

## Configuration

Install the project dependencies and add the following values to `.env`:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

For self-hosting, set `LANGFUSE_BASE_URL` to the deployment URL. Keep
`LANGFUSE_ENABLED=false` for autonomous local operation.

Run the application from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\tutor_bot\ui\app.py
```

The Observability page shows both sink statuses and performs an explicit Langfuse
authentication check when opened. This diagnostic check is not performed for every request.

## Trace model

One user operation is a trace. Retrieval, reranking/context selection, generation,
validation and evaluation are child observations. Generation observations contain
provider, model and token usage when returned by the provider. Source metadata contains
note and chunk identifiers, paths and ranking scores; full source documents are not sent.

The local service provides `record_feedback(...)` for future explicit user ratings.
There is no feedback UI yet, so the integration does not create implicit ratings.
