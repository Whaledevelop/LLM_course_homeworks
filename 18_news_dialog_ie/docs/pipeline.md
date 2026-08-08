# Pipeline

## Данные

`allenai/WildChat-1M` читается в streaming-режиме. Итоговая выборка содержит 200 уникальных англоязычных диалогов с источником `allenai/WildChat-1M`.

News filter использует score из независимых признаков: источник/атрибуция, news-like intent, событийная лексика, дата, reporting verb и capitalized phrases. Для принятия нужны score не ниже 4, сильный новостной сигнал и дополнительный признак события, даты или reporting verb. Все термины проверяются regex с границами слов.

До scoring исключаются явные programming/code, jailbreak, roleplay, explicit-content и advertising-generation prompts. Фильтр остаётся простым rule-based этапом и не использует LLM, spaCy или ML.

`--rebuild-dataset` удаляет только локальный dataset cache перед загрузкой. `--rebuild-cache` независимо очищает extraction cache.

## Gold-разметка

Из первых 20 выбранных диалогов создаётся annotation template. Разметка выполняется вручную через `annotation_app.py`. CSV содержит только `dialog_id,label,value`; reviewed-состояние и fingerprint шаблона хранятся отдельно в `annotation_progress.json`.

При смене template ID старые annotation-файлы сохраняются в backup и workspace сбрасывается. Диалог без сущностей может быть reviewed без строки в gold CSV. Benchmark допускается только после review всех 20 диалогов.

## Модели и benchmark

Основные профили:

| Профиль | Модель | Precision |
| --- | --- | --- |
| `mistral-fp16` | `mistralai/Mistral-7B-Instruct-v0.2` | FP16 |
| `mistral-int8` | та же модель | INT8 |
| `openchat-fp16` | `openchat/openchat-3.5-0106` | FP16 |
| `openchat-int8` | та же модель | INT8 |

`rules` и `spacy` доступны как optional baselines. Для каждого профиля измеряются throughput, mean/p95 latency, RAM, VRAM, precision, recall и F1 на batch size `1/2/4/8`.

Основная таблица записывается в `benchmark_results.csv`. Расширенные per-class метрики, ошибки, predictions и JSON-примеры сохраняются как вспомогательные материалы.
