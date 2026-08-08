# Извлечение сущностей и событий из новостных диалогов

Домашнее задание по треку B: 200 реальных новостных диалогов из `allenai/WildChat-1M`, 20 вручную размеченных gold-диалогов и сравнение двух локальных LLM в FP16/INT8. Извлекаются `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`.

## Что реализовано

- Воспроизводимый streaming-отбор WildChat по нескольким новостным признакам с исключением code, jailbreak, roleplay и рекламных prompts.
- Две LLM: `Mistral-7B-Instruct-v0.2` и `OpenChat-3.5-0106`, каждая в FP16 и INT8.
- Batch benchmark для размеров `1/2/4/8`, extraction cache и продолжение после CUDA OOM.
- Throughput, mean/p95 latency, RAM, VRAM, precision, recall и F1.
- Streamlit demo и отдельный инструмент полностью ручной gold-разметки.
- Rules и spaCy сохранены как необязательные baselines.

## Установка

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

INT8 через `bitsandbytes` рекомендуется запускать в Linux/WSL с NVIDIA CUDA. Веса моделей должны быть доступны через Hugging Face или локальный cache.

## Подготовка данных

Команда принудительно пересоздаёт выборку из 200 диалогов и шаблон для 20 ручных аннотаций:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 20 `
  --prepare-annotations `
  --rebuild-dataset
```

Если набор dialog ID изменился, существующие template/gold/progress сначала копируются в `data/annotation_backups`, после чего annotation workspace сбрасывается. При неизменном шаблоне прогресс сохраняется.

## Ручная gold-разметка

```powershell
streamlit run scripts/annotation_app.py
```

Добавление, удаление и отметка reviewed выполняются только вручную. Диалог без целевых сущностей следует отметить reviewed без добавления пустой CSV-строки. Gold сохраняется в `data/gold_annotations.csv`, прогресс — в `data/annotation_progress.json`.

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

Необязательные baselines:

```powershell
python scripts/run_demo.py --sample-size 200 --gold-size 20 --profiles rules spacy
```

## Demo и тесты

```powershell
streamlit run scripts/app.py
python -m pytest tests -q
```

Основные результаты сохраняются в `data/benchmark_results.csv`. Per-class метрики, ошибки, predictions и примеры JSON остаются вспомогательными артефактами для анализа.
