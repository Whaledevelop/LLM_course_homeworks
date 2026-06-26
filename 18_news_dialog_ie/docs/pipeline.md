# Pipeline

## Данные

Выбран трек B: новостные диалоги. Основной датасет - `allenai/WildChat-1M`. Он читается в streaming-режиме через Hugging Face `datasets`, чтобы не скачивать весь корпус. Фильтрация выполняется по новостным маркерам: `Reuters`, `BBC`, `CNN`, `election`, `minister`, `government`, `earthquake`, `market`, `court`, `reported` и похожим словам.

Для воспроизводимости добавлен fallback-набор из коротких новостных диалогов. Он используется, если библиотека `datasets` не установлена, нет доступа к Hugging Face или потоковая загрузка не успела набрать нужный размер.

## Схема извлечения

Целевые сущности:

| Label | Смысл |
| --- | --- |
| PERSON | политики, руководители, публичные лица |
| ORG | компании, агентства, институты |
| LOC | страны, города, регионы |
| DATE | даты и годы |
| EVENT | событие: meeting, earthquake, investigation, layoffs |
| IMPACT | последствия события |
| SOURCE | источник новости |

Отношение строится как компактная карточка события:

```json
{
  "event": "earthquake",
  "source": "AP News",
  "date": "April 3, 2024",
  "location": "Hualien",
  "impact": "damaging buildings"
}
```

## Модели

В проекте есть три режима:

| Режим | Назначение |
| --- | --- |
| `rules` | быстрый baseline, воспроизводимый на CPU |
| `spacy` | локальный NER для `PERSON/ORG/LOC/DATE` |
| `transformers` | локальная LLM для JSON IE |

LLM-профиль поддерживает full precision и quantized inference:

- full precision: `torch.float16` на GPU или `torch.float32` на CPU;
- quantized: `load_in_8bit=True` через `bitsandbytes`.

Для трека B можно использовать `Llama-2-7B`, `OpenChat-7B`, `Vicuna-7B` или `Mistral-7B-Instruct`. В CLI модель передается параметром `--llm-model`, поэтому один и тот же benchmark можно прогнать на нескольких локальных моделях.

## Prompt

LLM получает диалог и строгую JSON-схему:

```text
Extract news entities and events from the dialog.
Return only valid JSON matching this schema:
{"entities": [...], "events": [...], "relations": [...]}
```

Парсер извлекает первый JSON-объект из ответа. Если JSON невалиден, результат считается пустым. Это снижает риск падения batch-инференса на одном плохом ответе.

## Batch Processing

`ExtractionBenchmark` разбивает данные на batch-и:

- rule-based и spaCy обрабатывают список диалогов последовательно внутри batch;
- Transformers extractor передает список prompt-ов в `pipeline` с `batch_size`;
- результаты сохраняются в `data/cache/*.jsonl`;
- повторный запуск не пересчитывает уже обработанный профиль.

## Метрики

Измеряются:

- `docs_per_second`;
- `chars_per_second`;
- `estimated_tokens_per_second`;
- mean latency;
- p95 latency;
- peak RSS memory;
- precision/recall/F1 на контрольной разметке.

Качество считается по строгому совпадению пары `(label, normalized value)`. Для учебного benchmark это достаточно прозрачно: видно, какие классы извлекаются стабильно, а где нужны правила или LLM.

## CSV artifacts

Demo run additionally writes flat CSV files for review and notebook analysis:

- `data/gold_annotations.csv` - gold labels with `dialog_id`, `label`, `value`.
- `data/extraction_predictions.csv` - predicted entity/event rows with offsets and confidence.
- `data/benchmark_results.csv` - aggregate speed, memory and quality metrics, including estimated tokens/sec.
