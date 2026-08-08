# Извлечение сущностей и событий из новостных диалогов

Домашнее задание по треку B: 200 реальных новостных диалогов из `allenai/WildChat-1M`, 20 вручную размеченных gold-диалогов и сравнение двух локальных LLM в FP16/INT8. Извлекаются `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`.

## Pipeline

```text
WildChat stream
→ rule-based high-recall prefilter
→ OpenAI-compatible NEWS / NOT_NEWS classifier
→ 200 news dialogs
→ manual gold annotation for 20 dialogs
→ local Mistral/OpenChat IE benchmark
```

Classifier используется только при подготовке датасета. Он не является benchmark-моделью и не участвует в сравнении качества NER/IE.

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

Скопируйте `.env.example` в `.env` и укажите OpenAI-compatible classifier:

```dotenv
NEWS_CLASSIFIER_BASE_URL=http://localhost:8000/v1
NEWS_CLASSIFIER_API_KEY=
NEWS_CLASSIFIER_MODEL=classifier-model-name
```

`NEWS_CLASSIFIER_API_KEY` может быть пустым для локального endpoint. `.env` исключён из Git.

## Подготовка данных

Пересоздать выборку, сохранив существующий classifier cache:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 20 `
  --prepare-annotations `
  --rebuild-dataset
```

Повторно отправить кандидатов в classifier, удалив его cache:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 20 `
  --prepare-annotations `
  --rebuild-dataset `
  --rebuild-classifier-cache
```

Classifier cache хранится в `data/cache/news_classifier.jsonl`. Обычные `--rebuild-dataset` и `--rebuild-cache` его не удаляют.

## Ручная gold-разметка

```powershell
streamlit run scripts/annotation_app.py
```

Gold сохраняется в `data/gold_annotations.csv`, reviewed-прогресс — в `data/annotation_progress.json`. LLM, rules и spaCy не используются для pre-annotation.

## Основной benchmark

После завершения всех 20 reviewed-диалогов:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 20 `
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
