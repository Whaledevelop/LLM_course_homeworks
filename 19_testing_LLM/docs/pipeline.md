# Pipeline

## Задача

Цель домашней работы - показать, как LLM-приложение можно проверять автоматическими тестами и подключать к CI/CD quality gates. В качестве приложения используется простой локальный RAG/QA baseline, чтобы решение было воспроизводимым без OpenAI, Hugging Face token или другого внешнего ключа.

## Данные

Корпус задается в `scripts/dataset.py` и сохраняется в `data/corpus.jsonl`. Документы покрывают темы из лекции:

| Документ | Смысл |
| --- | --- |
| `ragas-faithfulness` | Проверка галлюцинаций и groundedness |
| `ragas-answer-relevance` | Соответствие ответа вопросу |
| `ragas-context-recall` | Полнота найденного контекста |
| `cicd-quality-gates` | Автоматический fail при просадке метрик |
| `golden-dataset` | Роль golden-примеров |
| `canary-prompts` | Быстрые safety/regression checks |
| `json-contracts` | Проверка строгого формата ответа |
| `monitoring-drift` | Наблюдаемость, latency, drift и feedback |

Golden-набор лежит в `tests/goldens.json`. Каждый пример содержит:

- `question` - вход пользователя;
- `reference` - эталонное ожидание;
- `expected_context_ids` - документы, которые должен вернуть retriever;
- `key_facts` - факты, которые должны быть поддержаны контекстом.

## RAG-приложение

`LocalRagApplication` выполняет три шага:

1. Токенизирует вопрос.
2. Считает overlap вопроса с каждым документом и выбирает top-k документов.
3. Формирует ответ только из фактов найденных документов.

Такой baseline выбран специально: он простой, быстрый и детерминированный. Для учебной CI/CD задачи важнее показать проверяемый контракт качества, чем подключать тяжелую локальную LLM.

## Ragas-совместимая оценка

Ragas обычно оценивает связку:

```text
question -> contexts -> answer -> reference
```

В проекте сохраняется та же схема данных и названия метрик:

- `faithfulness` - доля ключевых фактов, поддержанных найденным контекстом, с небольшим штрафом за слова ответа вне контекста;
- `answer_relevance` - overlap ответа с вопросом и эталонным reference;
- `context_recall` - доля ожидаемых документов, найденных retriever.

Зависимость `ragas` добавлена в `requirements.txt`, но demo run не требует внешнего judge LLM. Это позволяет запускать quality gates локально и в GitHub Actions без секретов.

## Quality gates

Пороги заданы в `scripts/evaluator.py`:

| Метрика | Порог |
| --- | ---: |
| Faithfulness | 0.70 |
| Answer relevance | 0.65 |
| Context recall | 0.75 |

`tests/test_quality_gates.py` запускает pipeline на всех goldens и проверяет средние метрики. Если retriever, generation logic или данные изменятся и качество упадет, pytest завершится ошибкой.

## CI/CD

Workflow `.github/workflows/llm-quality-gates.yml` делает стандартные шаги:

1. Checkout репозитория.
2. Установка Python.
3. Установка зависимостей из `19_testing_LLM/requirements.txt`.
4. Запуск `python -m pytest -q 19_testing_LLM/tests`.
5. Загрузка JSON/HTML отчетов как artifacts, если они были созданы demo run или тестами.

В production-проекте сюда можно добавить сравнение с baseline, canary prompts, budget по latency/cost и публикацию отчета в observability-систему.
