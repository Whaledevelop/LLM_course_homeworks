# Извлечение сущностей и событий из новостных диалогов

Домашнее задание по треку B: 200 реальных новостных диалогов из `allenai/WildChat-1M`, 10 вручную размеченных gold-диалогов и сравнение двух локальных LLM в FP16/INT8. Извлекаются `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`.

## Pipeline

```text
WildChat stream
→ rule-based high-recall prefilter
→ local Hugging Face Transformers NEWS / NOT_NEWS classifier
→ 200 news dialogs
→ manual gold annotation for 10 dialogs
→ local Mistral/OpenChat IE benchmark
```

Classifier `Qwen/Qwen2.5-0.5B-Instruct` используется только при подготовке датасета. Это небольшая instruction-tuned модель: веса скачиваются с Hugging Face Hub, inference выполняется локально на CPU или автоматически на CUDA. Classifier не является benchmark-моделью и не участвует в сравнении качества NER/IE.

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

Модель можно изменить через `--news-classifier-model`; CLI имеет приоритет над optional environment variable `NEWS_CLASSIFIER_MODEL`. Default batch size равен 8 и меняется через `--news-classifier-batch-size`.

## Подготовка данных

Пересоздать выборку, сохранив существующий classifier cache:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --prepare-annotations `
  --rebuild-dataset `
  --news-classifier-model Qwen/Qwen2.5-0.5B-Instruct `
  --news-classifier-batch-size 8
```

Повторно отправить кандидатов в classifier, удалив его cache:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --prepare-annotations `
  --rebuild-dataset `
  --rebuild-classifier-cache `
  --news-classifier-model Qwen/Qwen2.5-0.5B-Instruct
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
