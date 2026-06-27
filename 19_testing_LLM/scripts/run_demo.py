from __future__ import annotations

import argparse
from pathlib import Path

from dataset import load_corpus, write_corpus
from evaluator import load_goldens, run_evaluation, write_html_report, write_json_report
from rag_app import LocalRagApplication


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    corpus_path = data_dir / "corpus.jsonl"
    goldens_path = project_dir / "tests" / "goldens.json"

    if args.rebuild_data:
        documents = write_corpus(corpus_path)
    else:
        documents = load_corpus(corpus_path)

    app = LocalRagApplication(documents, top_k=args.top_k)
    goldens = load_goldens(goldens_path)
    summary, rows = run_evaluation(app, goldens)

    write_json_report(data_dir / "ragas_results.json", summary, rows)
    write_html_report(data_dir / "ragas_results.html", summary, rows)
    print_summary(summary)


def print_summary(summary) -> None:
    print(f"examples={summary.examples}")
    print(f"faithfulness={summary.faithfulness:.3f} threshold={summary.thresholds['faithfulness']:.3f}")
    print(f"answer_relevance={summary.answer_relevance:.3f} threshold={summary.thresholds['answer_relevance']:.3f}")
    print(f"context_recall={summary.context_recall:.3f} threshold={summary.thresholds['context_recall']:.3f}")
    print(f"pass_rate={summary.pass_rate:.3f}")
    print("reports=data/ragas_results.json,data/ragas_results.html")


if __name__ == "__main__":
    main()
