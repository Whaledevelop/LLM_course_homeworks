# Домашняя работа 19 — Автоматизация тестирования LLM и CI/CD

## Цель

Собрать небольшой RAG по Unity-документам, проверить его качество через Ragas, завернуть проверки в pytest и добавить CI/CD quality gate.

Главная цель домашки — не сам RAG, а автоматизированная проверка качества LLM/RAG-системы.

---

## Что требуется по домашке

1. Простое LLM-приложение.
2. 10–20 golden-примеров с эталонными ответами.
3. Оценка через Ragas: Faithfulness, Answer Relevance, Context Recall.
4. Автотесты с порогами качества.
5. CI/CD через GitHub Actions или описание пайплайна.
6. Сохранённый отчёт с результатами.
7. README с описанием метрик, порогов и запуска.
8. Скриншоты успешных и неуспешных тестов.

---

## Общая схема

```text
Unity-документы
      ↓
     RAG
      ↓
question → retrieval → contexts → LLM → answer
                         ↓
                       Ragas
                         ↓
 Faithfulness / Answer Relevance / Context Recall
                         ↓
                       pytest
                         ↓
                   quality gates
                         ↓
                  GitHub Actions
```

---

## Текущее состояние

### Уже сделано

- [x] Создан тестовый корпус из 6 документов по Unity.
- [x] Создан базовый каркас RAG.
- [x] Есть `app/config.py`.
- [x] Есть `app/rag.py`.
- [x] Есть `.env.example`.
- [x] Есть `requirements.txt`.

Документы:

- `gameobject_transform.md`
- `memory_gc.md`
- `monobehaviour_lifecycle.md`
- `physics.md`
- `scriptable_object.md`
- `update_methods.md`

### На чём остановились

Следующий шаг — настроить `.env`, подключить реальные LLM/embeddings и убедиться, что RAG локально отвечает на вопросы и возвращает найденный контекст.

---

## План реализации

### Этап 1. Проверить текущий RAG

**Ответственный: вручную + Codex для исправлений.**

Нужно:

- разобраться, какие переменные ожидает `.env.example`;
- выбрать LLM;
- выбрать embeddings;
- получить API-ключи и доступы;
- создать локальный `.env`;
- установить зависимости;
- запустить RAG;
- задать тестовый вопрос;
- убедиться, что возвращаются `question`, `answer`, `contexts`.

Пример вопроса:

```text
В чем разница между Update и FixedUpdate?
```

**Готово, когда:** RAG стабильно отвечает и использует документы из `data/documents`.

### Этап 2. Подготовить golden dataset

**Ответственный: содержимое — вручную/с ChatGPT; файл и код — Codex.**

Создать `tests/goldens.json` и подготовить 15 вопросов по существующим Unity-документам.

Пример:

```json
{
  "question": "Когда вызывается Start в MonoBehaviour?",
  "reference_answer": "Start вызывается перед первым Update, если компонент включён."
}
```

Вопросы должны покрывать разные документы и иметь однозначные эталонные ответы.

**Готово, когда:** есть 15 вручную проверенных golden-примеров.

### Этап 3. Подключить Ragas

**Ответственный: Codex.**

Evaluation pipeline для каждого golden-примера получает:

- question;
- answer;
- contexts;
- reference answer.

Используем обязательные метрики:

- **Faithfulness** — следует ли ответ из найденного контекста; основная проверка на галлюцинации.
- **Answer Relevance** — отвечает ли сгенерированный ответ на поставленный вопрос.
- **Context Recall** — нашёл ли retrieval информацию, необходимую для эталонного ответа.

**Готово, когда:** Ragas считает все три метрики на golden dataset.

### Этап 4. Сохранить отчёт Ragas

**Ответственный: Codex.**

Создать, например, `reports/ragas_results.json` с результатами по тестам и агрегированными значениями.

Пример:

```json
{
  "faithfulness": 0.87,
  "answer_relevancy": 0.91,
  "context_recall": 0.82
}
```

**Готово, когда:** после evaluation создаётся JSON-отчёт.

### Этап 5. Добавить quality gates

**Ответственный: пороги выбираем вручную, реализация — Codex.**

Стартовые учебные пороги:

| Метрика | Минимум |
|---|---:|
| Faithfulness | 0.70 |
| Answer Relevance | 0.70 |
| Context Recall | 0.70 |

Логика:

```text
score >= threshold → PASS
score < threshold  → FAIL
```

После первых реальных результатов пороги можно скорректировать.

### Этап 6. Добавить pytest

**Ответственный: Codex.**

Ragas evaluation должен вызываться из автотестов командой:

```bash
pytest
```

Тесты должны падать, если хотя бы одна обязательная агрегированная метрика ниже установленного порога.

**Готово, когда:** есть успешный локальный прогон `pytest`.

### Этап 7. Получить намеренно проваленный тест

**Ответственный: вручную + Codex при необходимости.**

Временно испортить конфигурацию или повысить threshold так, чтобы один quality gate не прошёл.

Например:

```text
Faithfulness = 0.81
Threshold = 0.90
→ FAIL
```

Сделать скриншот проваленного теста, затем вернуть нормальную конфигурацию.

**Готово, когда:** есть и PASS, и FAIL запуск.

### Этап 8. Настроить GitHub Actions

**Ответственный: workflow — Codex; GitHub Secrets и настройки — вручную.**

Создать `.github/workflows/llm-tests.yml`.

Pipeline:

```text
push / pull request
        ↓
checkout
        ↓
setup Python
        ↓
install requirements
        ↓
pytest + Ragas
        ↓
quality gate
        ↓
PASS / FAIL
```

Секреты API должны храниться в GitHub Actions Secrets, а не в репозитории.

**Готово, когда:** workflow запускается автоматически и может завершаться как успешно, так и с ошибкой quality gate.

### Этап 9. Собрать скриншоты

**Ответственный: вручную.**

Нужно сохранить:

1. Результаты Ragas с метриками.
2. Успешный локальный `pytest`.
3. Неуспешный локальный `pytest`.
4. Успешный GitHub Actions run.
5. Проваленный GitHub Actions run.

### Этап 10. Финализировать README

**Ответственный: Codex + ручная проверка.**

В финальном README оставить:

- краткое описание проекта;
- структуру основных папок;
- используемые метрики;
- выбранные thresholds;
- команды запуска;
- реальные результаты Ragas;
- описание GitHub Actions;
- объяснение провала quality gate;
- пути к отчётам.

Текущий большой план после завершения домашки можно сократить до итогового отчёта.

---

## Планируемая структура

```text
19_testing_LLM/
├── app/
│   ├── __init__.py
│   ├── config.py
│   └── rag.py
├── data/
│   └── documents/
│       ├── gameobject_transform.md
│       ├── memory_gc.md
│       ├── monobehaviour_lifecycle.md
│       ├── physics.md
│       ├── scriptable_object.md
│       └── update_methods.md
├── tests/
│   ├── goldens.json
│   ├── evaluate.py
│   └── test_ragas.py
├── reports/
│   └── ragas_results.json
├── .github/
│   └── workflows/
│       └── llm-tests.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## Разделение работы

| Задача | Кто делает |
|---|---|
| Регистрации, API-ключи, облачные настройки | Вручную |
| `.env` с реальными секретами | Вручную |
| Локальный запуск команд | Вручную |
| GitHub Secrets | Вручную |
| Скриншоты | Вручную |
| RAG и Python-код | Codex |
| Ragas integration | Codex |
| pytest | Codex |
| GitHub Actions YAML | Codex |
| Структура JSON/отчётов | Codex |
| Промпты для Codex | ChatGPT |
| Разбор ошибок и объяснение шагов | ChatGPT |

---

## Финальный чеклист сдачи

- [ ] RAG запускается и отвечает по Unity-документам.
- [ ] Есть 10–20 goldens.
- [ ] Работает Faithfulness.
- [ ] Работает Answer Relevance.
- [ ] Работает Context Recall.
- [ ] Результаты сохраняются в JSON/HTML.
- [ ] Есть quality thresholds.
- [ ] Есть pytest.
- [ ] Есть успешный тестовый прогон.
- [ ] Есть намеренно проваленный тестовый прогон.
- [ ] Есть GitHub Actions либо подробное описание CI/CD.
- [ ] Есть скриншоты результатов.
- [ ] README содержит реальные результаты и их интерпретацию.
