# Автоматизация тестирования LLM и CI/CD quality gates

Домашняя работа по лекции 19 реализована как локальный Python-проект без внешних API-ключей. В проекте есть простое RAG/QA-приложение, golden-набор из 12 примеров, Ragas-совместимая оценка `faithfulness`, `answer_relevance`, `context_recall`, pytest-гейты и GitHub Actions workflow.

## Что сделано

- Подготовлен небольшой корпус документов по тестированию LLM, RAGAS, canary prompts, JSON contracts, monitoring и CI/CD.
- Реализован локальный RAG baseline: keyword retrieval + grounded answer из найденных фактов.
- Добавлены goldens в `tests/goldens.json`.
- Реализован evaluator с метриками:
  - `faithfulness` >= 0.70;
  - `answer_relevance` >= 0.65;
  - `context_recall` >= 0.75.
- Добавлен `pytest`-тест, который падает при просадке quality gates.
- Добавлен CI workflow `.github/workflows/llm-quality-gates.yml`.
- Demo run сохраняет результаты в `data/ragas_results.json` и `data/ragas_results.html`.

## Установка

Рекомендуется Python 3.10-3.12. В CI используется Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Быстрый запуск

```powershell
python scripts/run_demo.py --rebuild-data
```

Проверка quality gates:

```powershell
python -m pytest -q
```

## Структура

- `scripts/dataset.py` - локальный корпус документов.
- `scripts/rag_app.py` - простой RAG/QA pipeline.
- `scripts/evaluator.py` - Ragas-совместимые метрики, JSON/HTML отчеты и thresholds.
- `scripts/run_demo.py` - воспроизводимый demo run.
- `tests/goldens.json` - 12 golden-примеров.
- `tests/test_quality_gates.py` - pytest quality gates.
- `data/corpus.jsonl` - корпус, генерируется demo run.
- `data/ragas_results.json` - JSON-отчет, генерируется demo run.
- `data/ragas_results.html` - HTML-отчет, генерируется demo run.
- `docs/pipeline.md` - описание pipeline.
- `docs/result_analyze.md` - результаты, метрики, trade-offs и выводы.

## CI/CD

Workflow запускается на push и pull request:

```yaml
python -m pip install -r 19_testing_LLM/requirements.txt
python -m pytest -q 19_testing_LLM/tests
```

Если средние метрики ниже порогов, job падает. В реальном production-проекте JSON/HTML отчеты можно публиковать как CI artifacts и сравнивать с baseline предыдущего релиза.
