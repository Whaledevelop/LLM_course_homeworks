# Извлечение сущностей и событий из новостных диалогов

Домашнее задание по уроку 18 выполнено по треку B: извлечение из новостных диалогов. Проект загружает и фильтрует `allenai/WildChat-1M`, выделяет новостные диалоги, извлекает сущности `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`, строит простые отношения и измеряет качество/скорость.

## Что сделано

- Подготовлен потоковый загрузчик WildChat-1M с fallback-датасетом для воспроизводимого запуска без скачивания 1M диалогов.
- Реализован rule-based IE baseline для новостных диалогов.
- Добавлена опциональная интеграция spaCy NER.
- Добавлен опциональный локальный LLM extractor через Hugging Face Transformers.
- Поддержан режим full precision и quantized inference для локальной модели.
- Реализованы batch processing, JSONL cache, throughput/latency/RSS benchmark.
- Реализована оценка precision/recall/F1 на небольшой размеченной контрольной выборке.
- Добавлены CSV-артефакты с gold-разметкой и предсказанными сущностями для проверки качества.
- Добавлено Streamlit-приложение для демонстрации извлечения.

## Установка

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Для spaCy-профиля:

```powershell
python -m spacy download en_core_web_sm
```

## Быстрый запуск

```powershell
python scripts/run_demo.py --sample-size 100 --batch-size 16 --rebuild-cache
```

Запуск со spaCy:

```powershell
python scripts/run_demo.py --sample-size 100 --batch-size 16 --with-spacy --rebuild-cache
```

Запуск локальной LLM в full precision:

```powershell
python scripts/run_demo.py --sample-size 20 --batch-size 2 --llm-model mistralai/Mistral-7B-Instruct-v0.2 --rebuild-cache
```

Запуск quantized-профиля:

```powershell
python scripts/run_demo.py --sample-size 20 --batch-size 2 --llm-model mistralai/Mistral-7B-Instruct-v0.2 --llm-quantized --rebuild-cache
```

На Windows `bitsandbytes` может быть недоступен. В таком случае quantized-профиль лучше запускать в WSL/Linux или заменить модель на локальный GGUF через `llama.cpp`.

## Demo UI

```powershell
streamlit run scripts/app.py
```

## Структура

- `scripts/dataset.py` - загрузка WildChat-1M, фильтрация новостных диалогов, fallback-примеры.
- `scripts/extractors.py` - rule-based, spaCy и Transformers extractors.
- `scripts/benchmark.py` - batching, cache, latency, throughput, RSS.
- `scripts/evaluation.py` - контрольная разметка и precision/recall/F1.
- `scripts/run_demo.py` - воспроизводимый CLI-сценарий.
- `scripts/app.py` - Streamlit-приложение.
- `docs/pipeline.md` - описание пайплайна.
- `docs/result_analyze.md` - анализ результатов и trade-offs.
- `data/benchmark_results.csv` - генерируется запуском demo.
- `data/gold_annotations.csv` - контрольная CSV-разметка для расчета precision/recall/F1.
- `data/extraction_predictions.csv` - плоская CSV-таблица извлеченных сущностей и событий.
- `data/extractions.json` - примеры извлечений.

## Текущий воспроизводимый прогон

В репозитории сохранен быстрый CPU-прогон:

```powershell
python scripts/run_demo.py --sample-size 100 --batch-size 16 --rebuild-cache
```

Он формирует:

- `data/news_dialogs.jsonl` - 100 demo-диалогов.
- `data/gold_annotations.csv` - 840 строк контрольной разметки.
- `data/extraction_predictions.csv` - 920 строк предсказаний.
- `data/benchmark_results.csv` - throughput, latency, estimated tokens/sec, RSS и precision/recall/F1.

Сохраненный benchmark содержит `rules` baseline. LLM-профили запускаются теми же скриптами через `--llm-model`, но требуют локально доступной 7B-модели и GPU/достаточной RAM.

## Вывод

Для учебной задачи baseline на регулярных выражениях дает быстрый и прозрачный IE-пайплайн. spaCy повышает покрытие `PERSON/ORG/LOC/DATE`, но не решает `IMPACT` и доменные события без дополнительных правил. Локальная LLM лучше подходит для сложных событий и отношений, но требует GPU/quantization и строгого JSON-парсинга.
