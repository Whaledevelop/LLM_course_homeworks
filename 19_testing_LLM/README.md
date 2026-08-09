# Домашняя работа 19 — автоматизация тестирования LLM и CI/CD

## О проекте

Домашняя работа реализует небольшой **RAG-сервис по документации Unity** и автоматическую проверку его качества.

Пайплайн:

```text
вопрос → embeddings → поиск контекстов в Chroma → ответ LLM
      → Ragas evaluation → pytest quality gates → GitHub Actions
```

Для проверки подготовлено **15 golden-примеров** по 6 Unity-темам. Качество оценивается метриками **Faithfulness**, **Answer Relevancy** и **Context Recall**.

## Структура

```text
19_testing_LLM/
├── app/
│   ├── config.py              # конфигурация из .env
│   ├── embeddings.py          # Yandex Embeddings
│   └── rag.py                 # RAG-пайплайн
├── data/
│   ├── documents/             # 6 исходных документов по Unity
│   └── reports/               # сохранённые результаты Ragas
├── tests/
│   ├── goldens.json           # 15 эталонных question/answer
│   ├── evaluate.py            # запуск Ragas evaluation
│   └── test_ragas.py          # pytest quality gates
├── screenshots/               # скриншоты результатов и CI
├── .env.example
├── requirements.txt
└── README.md
```

GitHub Actions workflow находится в корне репозитория:

```text
.github/workflows/llm-tests.yml
```

## Модели и данные

Используются:

- **LLM:** Qwen3-235B через Yandex AI Studio OpenAI-compatible API;
- **Embeddings:** `text-embeddings-v2` через Yandex AI Studio;
- **Vector Store:** Chroma;
- **Evaluation:** Ragas;
- **Quality Gates:** pytest;
- **CI:** GitHub Actions.

Документы находятся в `data/documents/` и покрывают:

- Update / FixedUpdate / LateUpdate;
- lifecycle MonoBehaviour;
- Unity Physics;
- GameObject и Transform;
- память и Garbage Collector;
- ScriptableObject.

## Запуск

Требуется Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Создать `.env` на основе `.env.example` и указать API-ключи, модель и Yandex Folder ID.

Запуск RAG:

```powershell
python -m app.rag
```

Запуск полной оценки:

```powershell
python -m tests.evaluate
```

Запуск quality gates:

```powershell
pytest
```

## Golden dataset и quality gates

`tests/goldens.json` содержит **15 вопросов с эталонными ответами**. Они используются для автоматической оценки RAG.

Проверяются три средние метрики:

| Метрика | Базовый порог |
|---|---:|
| Faithfulness | 0.70 |
| Answer Relevancy | 0.70 |
| Context Recall | 0.70 |

Если хотя бы одна метрика ниже порога, pytest завершается с ошибкой и CI получает статус **FAIL**.

Порог Faithfulness также можно переопределить через `FAITHFULNESS_THRESHOLD`. Это используется для демонстрации срабатывания quality gate.

## Финальные результаты

Финальный GitHub Actions прогон на 15 golden-примерах:

| Метрика | Результат | Порог | Статус |
|---|---:|---:|---|
| Faithfulness | **0.981** | 0.70 | PASS |
| Answer Relevancy | **0.824** | 0.70 | PASS |
| Context Recall | **1.000** | 0.70 | PASS |

Все три quality gate успешно пройдены:

```text
3 passed
```

Небольшие различия значений между отдельными прогонами допустимы: Ragas использует LLM-as-a-Judge, поэтому evaluation не является полностью детерминированным.

## Где смотреть результаты

Основные результаты собраны в двух местах:

```text
data/reports/
```

- `ragas_results_final.json` — **финальный результат**, полученный из успешного GitHub Actions workflow;
- `ragas_results_test.json` — результат тестового/предыдущего прогона;
- внутри JSON находятся результаты каждого golden-примера и итоговый блок `averages`.

```text
screenshots/
```

Содержит скриншоты для проверки домашней работы: успешный CI pipeline, итоговые Ragas-метрики, Secrets/Variables и демонстрацию срабатывания quality gate.

Кроме того, каждый GitHub Actions запуск загружает `ragas_results.json` как artifact **`ragas-results`**.

## GitHub Actions

Workflow поддерживает:

```text
push
pull_request
workflow_dispatch
```

В GitHub настроены:

- Secrets: `OPENAI_API_KEY`, `EMBEDDING_API_KEY`;
- Variables: `LLM_MODEL`, `FAITHFULNESS_THRESHOLD`.

CI автоматически устанавливает зависимости, запускает pytest/Ragas, проверяет quality gates и сохраняет JSON-отчёт как artifact.
