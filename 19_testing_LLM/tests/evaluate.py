import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

from app.config import PROJECT_ROOT, Settings
from app.embeddings import YandexAIStudioEmbeddings
from app.rag import RagApplication


GOLDENS_PATH = PROJECT_ROOT / "tests" / "goldens.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "ragas_results.json"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_recall")


class _RagasYandexEmbeddings(BaseRagasEmbedding):
    def __init__(self, embeddings: YandexAIStudioEmbeddings) -> None:
        super().__init__()
        self._embeddings = embeddings

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self._embeddings.embed_query(text)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text, **kwargs)


class RagasEvaluator:
    def __init__(self, settings: Settings) -> None:
        self._rag = RagApplication(settings)
        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        evaluator_llm = llm_factory(settings.llm_model, client=client)
        evaluator_embeddings = _RagasYandexEmbeddings(
            YandexAIStudioEmbeddings(
                api_key=settings.embedding_api_key,
                folder_id=settings.yandex_folder_id,
                model=settings.embedding_model,
            )
        )
        self._faithfulness = Faithfulness(llm=evaluator_llm)
        self._answer_relevancy = AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        self._context_recall = ContextRecall(llm=evaluator_llm)

    def evaluate(self, goldens: list[dict[str, str]]) -> dict[str, Any]:
        samples = []
        for index, golden in enumerate(goldens, start=1):
            print(f"[{index}/{len(goldens)}] {golden['question']}")
            rag_result = self._rag.ask(golden["question"])
            scores = self._calculate_scores(
                question=golden["question"],
                answer=rag_result["answer"],
                contexts=rag_result["contexts"],
                reference_answer=golden["reference_answer"],
            )
            samples.append(
                {
                    "question": golden["question"],
                    "reference_answer": golden["reference_answer"],
                    "answer": rag_result["answer"],
                    "contexts": rag_result["contexts"],
                    "scores": scores,
                }
            )
            print(
                "  "
                + ", ".join(
                    f"{name}={score:.3f}" for name, score in scores.items()
                )
            )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(samples),
            "averages": {
                name: sum(sample["scores"][name] for sample in samples)
                / len(samples)
                for name in METRIC_NAMES
            },
            "samples": samples,
        }

    def _calculate_scores(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
    ) -> dict[str, float]:
        return asyncio.run(
            self._calculate_scores_async(
                question=question,
                answer=answer,
                contexts=contexts,
                reference_answer=reference_answer,
            )
        )

    async def _calculate_scores_async(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
    ) -> dict[str, float]:
        faithfulness, answer_relevancy, context_recall = await asyncio.gather(
            self._faithfulness.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            ),
            self._answer_relevancy.ascore(
                user_input=question,
                response=answer,
            ),
            self._context_recall.ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=reference_answer,
            ),
        )

        return {
            "faithfulness": float(faithfulness.value),
            "answer_relevancy": float(answer_relevancy.value),
            "context_recall": float(context_recall.value),
        }


def load_goldens(path: Path = GOLDENS_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        goldens = json.load(file)

    return goldens


def save_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    evaluator = RagasEvaluator(Settings.from_env())
    report = evaluator.evaluate(load_goldens())
    save_report(report)
    print(f"\nОтчет сохранен: {REPORT_PATH}")
    for name, score in report["averages"].items():
        print(f"{name}: {score:.3f}")


if __name__ == "__main__":
    main()
