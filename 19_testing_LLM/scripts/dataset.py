from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    facts: list[str]


DOCUMENTS = [
    Document(
        id="ragas-faithfulness",
        title="Faithfulness gate",
        text=(
            "Faithfulness checks whether an answer is grounded in retrieved context. "
            "A high faithfulness score means the answer does not add unsupported facts."
        ),
        facts=[
            "Faithfulness checks whether an answer is grounded in retrieved context",
            "A high faithfulness score means the answer does not add unsupported facts",
        ],
    ),
    Document(
        id="ragas-answer-relevance",
        title="Answer relevance gate",
        text=(
            "Answer relevance measures whether the generated answer addresses the user question. "
            "It should be used with golden examples because there may be several valid phrasings."
        ),
        facts=[
            "Answer relevance measures whether the generated answer addresses the user question",
            "It should be used with golden examples",
        ],
    ),
    Document(
        id="ragas-context-recall",
        title="Context recall gate",
        text=(
            "Context recall measures whether the retriever returned the evidence needed for the reference answer. "
            "Low context recall usually points to retrieval, chunking, or indexing problems."
        ),
        facts=[
            "Context recall measures whether the retriever returned the evidence needed for the reference answer",
            "Low context recall usually points to retrieval, chunking, or indexing problems",
        ],
    ),
    Document(
        id="cicd-quality-gates",
        title="CI/CD quality gates",
        text=(
            "CI/CD quality gates run automated checks before release. "
            "The pipeline should fail when LLM quality metrics fall below configured thresholds."
        ),
        facts=[
            "CI/CD quality gates run automated checks before release",
            "The pipeline should fail when LLM quality metrics fall below configured thresholds",
        ],
    ),
    Document(
        id="golden-dataset",
        title="Golden dataset",
        text=(
            "Golden examples contain inputs and expected answers or key facts. "
            "They are used for regression testing because LLM outputs are not fully deterministic."
        ),
        facts=[
            "Golden examples contain inputs and expected answers or key facts",
            "They are used for regression testing because LLM outputs are not fully deterministic",
        ],
    ),
    Document(
        id="canary-prompts",
        title="Canary prompts",
        text=(
            "Canary prompts are a small fixed set of risky checks. "
            "They help detect prompt leakage, PII leakage, jailbreak regressions, and broken output contracts."
        ),
        facts=[
            "Canary prompts are a small fixed set of risky checks",
            "They help detect prompt leakage, PII leakage, jailbreak regressions, and broken output contracts",
        ],
    ),
    Document(
        id="json-contracts",
        title="JSON contract tests",
        text=(
            "JSON contract tests validate that the model response follows the required schema. "
            "Contract tests are useful when the product expects strict machine-readable output."
        ),
        facts=[
            "JSON contract tests validate that the model response follows the required schema",
            "Contract tests are useful when the product expects strict machine-readable output",
        ],
    ),
    Document(
        id="monitoring-drift",
        title="Monitoring and drift",
        text=(
            "Production monitoring tracks quality, latency, format errors, and drift. "
            "Human feedback and sampled outputs help decide whether a model or prompt changed behavior."
        ),
        facts=[
            "Production monitoring tracks quality, latency, format errors, and drift",
            "Human feedback and sampled outputs help decide whether a model or prompt changed behavior",
        ],
    ),
]


def write_corpus(path: Path) -> list[Document]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for document in DOCUMENTS:
            file.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")

    return DOCUMENTS


def load_corpus(path: Path) -> list[Document]:
    if not path.exists():
        return write_corpus(path)

    documents = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            payload = json.loads(line)
            documents.append(Document(**payload))

    return documents
