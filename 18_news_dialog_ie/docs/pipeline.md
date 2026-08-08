# Pipeline

## Двухступенчатая подготовка данных

`allenai/WildChat-1M` читается в streaming-режиме до получения 200 уникальных диалогов, принятых как `NEWS`.

Stage 1 — дешёвый high-recall prefilter. Он оставляет английские тексты с news-source, news-intent, event или reporting признаками и исключает явные code, jailbreak, roleplay, explicit и advertising prompts. Его задача — сократить число HTTP-вызовов, а не принимать финальное решение.

Stage 2 — отдельный OpenAI-compatible classifier. Он получает полный диалог при `temperature=0` и возвращает строго `NEWS` или `NOT_NEWS`. Любой другой ответ считается `INVALID` и не попадает в dataset.

Classifier применяется только к подготовке данных. Основной IE benchmark по-прежнему сравнивает локальные Mistral/OpenChat в FP16/INT8.

## Cache и статистика

Результаты Stage 2 сохраняются в `data/cache/news_classifier.jsonl` по ключу из hash текста, модели и версии prompt. Cache хранит `NEWS`, `NOT_NEWS` и `INVALID`. Сетевые и schema errors не записываются.

- `--rebuild-dataset` пересоздаёт dataset, сохраняя classifier cache.
- `--rebuild-cache` очищает только extraction cache.
- `--rebuild-classifier-cache` явно удаляет cache Stage 2.

`dataset_stats.json` содержит число просмотренных строк, прошедших Stage 1, классифицированных candidates, `NEWS`, `NOT_NEWS`, `INVALID`, cache hits и имя classifier-модели.

## Gold и benchmark

Из первых 20 итоговых NEWS-диалогов создаётся template для полностью ручной разметки. CSV содержит `dialog_id,label,value`; reviewed-состояние хранится отдельно.

Основные профили benchmark:

| Профиль | Модель | Precision |
| --- | --- | --- |
| `mistral-fp16` | `mistralai/Mistral-7B-Instruct-v0.2` | FP16 |
| `mistral-int8` | та же модель | INT8 |
| `openchat-fp16` | `openchat/openchat-3.5-0106` | FP16 |
| `openchat-int8` | та же модель | INT8 |

Для batch size `1/2/4/8` измеряются throughput, mean/p95 latency, RAM, VRAM, precision, recall и F1. Rules/spaCy остаются optional baselines.
