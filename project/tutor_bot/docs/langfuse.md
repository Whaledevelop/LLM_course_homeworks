# Langfuse observability

Tutor Bot sends observability events directly to Langfuse. It does not keep a local
observability event store or dashboard.

## Configuration

Install the project dependencies and add the following values to `.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

For self-hosting, set `LANGFUSE_BASE_URL` to the deployment URL. Valid Langfuse
credentials are required to run LLM-backed features.

Run the application from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\tutor_bot\ui\app.py
```

## Trace model

One user operation is a trace. Retrieval, reranking/context selection, generation,
validation and evaluation are child observations. Generation observations contain
provider, model and token usage when returned by the provider. Source metadata contains
note and chunk identifiers, paths and ranking scores; full source documents are not sent.

The application service provides `record_feedback(...)` for future explicit user ratings.
There is no feedback UI yet, so the integration does not create implicit ratings.
