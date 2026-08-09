# Домашняя работа 19 — тестирование LLM и CI/CD

## О проекте

Проект представляет собой небольшой RAG по шести учебным документам о Unity. Пользователь задаёт вопрос, Chroma находит релевантные фрагменты, а генеративная модель формирует ответ только по найденному контексту.

Качество системы проверяется на 15 заранее подготовленных golden-примерах с помощью Ragas, pytest и GitHub Actions.

```text
вопрос → поиск контекстов → ответ LLM → Ragas → pytest → quality gate
```

## Структура

```text
19_testing_LLM/
├── app/
│   ├── config.py
│   ├── embeddings.py
│   └── rag.py
├── data/
│   └── documents/
│       ├── gameobject_transform.md
│       ├── memory_gc.md
│       ├── monobehaviour_lifecycle.md
│       ├── physics.md
│       ├── scriptable_object.md
│       └── update_methods.md
├── reports/
│   └── ragas_results.json
├── tests/
│   ├── evaluate.py
│   ├── goldens.json
│   └── test_ragas.py
├── .env.example
├── requirements.txt
└── README.md
```

GitHub Actions workflow расположен в корне общего репозитория: `.github/workflows/llm-tests.yml`.

## Настройка

Требуется Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Создайте `.env` на основе `.env.example`:

```dotenv
OPENAI_API_KEY=your_llm_api_key
OPENAI_BASE_URL=https://ai.api.cloud.yandex.net/v1
LLM_MODEL=your_chat_model

EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_MODEL=text-embeddings-v2
YANDEX_FOLDER_ID=your_folder_id

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVED_CHUNKS=3
```

`.env` содержит секреты и не должен попадать в репозиторий.

## Запуск RAG

```powershell
python -m app.rag
```

Приложение выводит сгенерированный ответ и найденные фрагменты документов.

## Golden dataset

Файл `tests/goldens.json` содержит 15 однозначных вопросов и эталонных ответов. Примеры покрывают все шесть тем:

| Тема | Вопросов |
|---|---:|
| Методы обновления | 3 |
| Жизненный цикл MonoBehaviour | 3 |
| Физика Unity | 3 |
| GameObject и Transform | 2 |
| Память и Garbage Collector | 2 |
| ScriptableObject | 2 |

## Метрики Ragas

Используются три обязательные метрики:

- **Faithfulness** — какая доля утверждений ответа подтверждается найденными контекстами. Низкое значение указывает на возможные галлюцинации.
- **Answer Relevance** — насколько ответ соответствует смыслу вопроса. Метрика не проверяет фактическую правильность отдельно.
- **Context Recall** — какая доля фактов эталонного ответа присутствует в найденных контекстах. Метрика показывает качество retrieval.

Полная оценка запускается командой:

```powershell
python -m tests.evaluate
```

Для каждого golden-примера сохраняются вопрос, эталонный ответ, ответ RAG, найденные контексты и три оценки. Итоговый отчёт записывается в `reports/ragas_results.json`.

## Результаты

Последний полный прогон на 15 примерах:

| Метрика | Среднее значение | Порог | Результат |
|---|---:|---:|---|
| Faithfulness | 0.967 | 0.70 | PASS |
| Answer Relevance | 0.818 | 0.70 | PASS |
| Context Recall | 1.000 | 0.70 | PASS |

Интерпретация:

- `Faithfulness = 0.967` означает, что ответы почти полностью основаны на найденных документах.
- `Answer Relevance = 0.818` показывает, что ответы в среднем хорошо соответствуют вопросам, хотя отдельные формулировки оцениваются ниже.
- `Context Recall = 1.000` означает, что retrieval нашёл контексты со всеми фактами, необходимыми для эталонных ответов.

Ragas использует LLM как проверяющую модель, поэтому значения могут немного изменяться между запусками.

## Pytest и quality gates

Для всех трёх средних метрик установлен учебный порог `0.70`.

```powershell
pytest
```

В пределах одного тестового сеанса Ragas evaluation запускается один раз. Затем три теста независимо проверяют средние значения. Если хотя бы одна метрика ниже порога, pytest завершается с ошибкой.

Последний полный локальный запуск:

```text
3 passed in 342.16s
```

Пороги можно временно переопределять переменными окружения:

- `FAITHFULNESS_THRESHOLD`;
- `ANSWER_RELEVANCY_THRESHOLD`;
- `CONTEXT_RECALL_THRESHOLD`.

### Намеренный FAIL

Для быстрой демонстрации quality gate можно использовать уже сохранённый отчёт и заведомо недостижимый порог:

```powershell
$env:RAGAS_USE_EXISTING_REPORT="1"
$env:FAITHFULNESS_THRESHOLD="1.01"
pytest
```

Полученный результат:

```text
faithfulness ниже порога: 0.967 < 1.010
1 failed, 2 passed
```

Этот режим предназначен только для демонстрации FAIL. Обычный запуск `pytest` не читает готовый отчёт, а выполняет свежую сетевую оценку.

После демонстрации удалите временные переменные:

```powershell
Remove-Item Env:RAGAS_USE_EXISTING_REPORT
Remove-Item Env:FAITHFULNESS_THRESHOLD
```

## GitHub Actions

Workflow запускается при изменениях в `19_testing_LLM`:

```text
push / pull request / ручной запуск
                ↓
        установка Python 3.12
                ↓
        установка зависимостей
                ↓
        pytest + свежая Ragas-оценка
                ↓
           PASS или FAIL
                ↓
 загрузка ragas_results.json как artifact
```

В `Settings → Secrets and variables → Actions` нужно добавить:

| Тип | Имя | Значение |
|---|---|---|
| Secret | `OPENAI_API_KEY` | API-ключ генеративной модели |
| Secret | `EMBEDDING_API_KEY` | API-ключ Yandex Embeddings |
| Variable | `LLM_MODEL` | URI генеративной модели из локального `.env` |

Workflow не использует сохранённый отчёт и всегда выполняет свежую оценку. JSON-отчёт загружается как artifact даже при провале quality gate.

## Что осталось для сдачи

Код приложения и автоматизация завершены. После добавления GitHub Secrets остаётся:

- запустить успешный GitHub Actions workflow;
- при необходимости временно задать завышенный порог для проваленного workflow;
- сделать скриншоты результатов Ragas, локальных PASS/FAIL и GitHub Actions PASS/FAIL.

## Финальный чеклист

- [x] RAG отвечает по Unity-документам.
- [x] Подготовлены 15 golden-примеров.
- [x] Работают Faithfulness, Answer Relevance и Context Recall.
- [x] JSON-отчёт содержит подробные и агрегированные результаты.
- [x] Настроены quality thresholds.
- [x] Добавлены pytest-тесты.
- [x] Получены локальные PASS и намеренный FAIL.
- [x] Добавлен GitHub Actions workflow.
- [x] README содержит команды, результаты и интерпретацию.
- [ ] Добавлены GitHub Secrets и выполнен workflow.
- [ ] Собраны скриншоты для сдачи.
