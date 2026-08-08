# Извлечение сущностей и событий из новостных диалогов

Домашнее задание по треку B: 200 реальных новостных диалогов из `allenai/WildChat-1M`, 10 вручную размеченных gold-диалогов и сравнение двух локальных LLM в FP16/INT8. Извлекаются `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`.

## Pipeline

```text
WildChat stream
→ rule-based high-recall prefilter
→ extract initial user intent without Assistant responses
→ truncate classifier input to 1024 tokens by default
→ prompt-injection-resistant Qwen3-1.7B NEWS / NOT_NEWS classifier
→ 200 news dialogs
→ manual gold annotation for 10 dialogs
→ local Mistral/OpenChat IE benchmark
```

Classifier `Qwen/Qwen3-1.7B` используется только при подготовке датасета. Это современная instruction-following модель примерно на 2B параметров: веса скачиваются с Hugging Face Hub, inference выполняется локально на CPU или автоматически на CUDA в non-thinking режиме. Модель загружается стандартными `AutoTokenizer`/`AutoModelForCausalLM` без `trust_remote_code`. Classifier не является benchmark-моделью и не участвует в сравнении качества NER/IE.

Основной benchmark включает:

- `mistral-fp16` и `mistral-int8`;
- `openchat-fp16` и `openchat-int8`;
- batch size `1/2/4/8`;
- throughput, latency, RAM, VRAM, precision, recall и F1.

Rules и spaCy доступны только как необязательные baselines.

## Установка

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Модель можно изменить через `--news-classifier-model`; CLI имеет приоритет над optional environment variable `NEWS_CLASSIFIER_MODEL`. Безопасный для RTX 2060 SUPER default batch size равен 4 и меняется через `--news-classifier-batch-size`. Лимит classifier input по умолчанию равен 1024 токенам и настраивается через `--news-classifier-max-input-tokens`. Опциональный `--news-classifier-sanity-check` проверяет classifier на реалистичных примерах WildChat и останавливает сканирование, если accuracy ниже 90%.

## Подготовка данных

Пересоздать выборку, сохранив существующий classifier cache:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --prepare-annotations `
  --rebuild-dataset `
  --news-classifier-model Qwen/Qwen3-1.7B `
  --news-classifier-batch-size 4 `
  --news-classifier-max-input-tokens 1024 `
  --news-classifier-sanity-check
```

Повторно отправить кандидатов в classifier, удалив его cache:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --prepare-annotations `
  --rebuild-dataset `
  --rebuild-classifier-cache `
  --news-classifier-model Qwen/Qwen3-1.7B `
  --news-classifier-sanity-check
```

Classifier загружается один раз и обрабатывает Stage 1 candidates пачками. Cache хранится в `data/cache/news_classifier.jsonl`; `--rebuild-dataset` и `--rebuild-cache` его не удаляют.

Во время подготовки CLI печатает результат классификации каждого Stage 1 candidate, короткий однострочный preview, отметку `[CACHE]` для повторно использованных ответов и общую статистику сканирования каждые 1000 строк WildChat. После сбора выборки выводятся итоговые счётчики и пути к dataset/template.

## Ручная gold-разметка

```powershell
streamlit run scripts/annotation_app.py
```

Gold сохраняется в `data/gold_annotations.csv`, reviewed-прогресс — в `data/annotation_progress.json`. LLM, rules и spaCy не используются для pre-annotation.

## Основной benchmark

После завершения всех 10 reviewed-диалогов:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --batch-sizes 1 2 4 8 `
  --profiles mistral-fp16 mistral-int8 openchat-fp16 openchat-int8 `
  --rebuild-cache
```

INT8 через `bitsandbytes` рекомендуется запускать в Linux/WSL с NVIDIA CUDA. Веса benchmark-моделей должны быть доступны через Hugging Face или локальный cache.

## Demo и тесты

```powershell
streamlit run scripts/app.py
python -m pytest tests -q
```

Filtering statistics сохраняются в `data/dataset_stats.json`. Основные результаты benchmark записываются в `data/benchmark_results.csv`; остальные CSV и JSON используются как вспомогательные артефакты анализа.
