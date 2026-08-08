# Извлечение сущностей и событий из новостных диалогов

Домашнее задание по треку B. Проект формирует уникальную англоязычную выборку `allenai/WildChat-1M`, извлекает `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE` и сравнивает rules, spaCy и две локальные 7B LLM в FP16/INT8.

## Реализовано

- Streaming-загрузка WildChat с проверкой языка, источника и уникальности текста.
- Синтетический fallback, разрешённый только явным флагом для smoke-тестов.
- Профили `rules`, `spacy`, `mistral-fp16`, `mistral-int8`, `openchat-fp16`, `openchat-int8`.
- Chat templates, строгая JSON-схема, валидация labels и учёт ошибок парсинга.
- Batch benchmark для размеров `1/2/4/8` с продолжением после CUDA OOM.
- Fingerprint cache по модели, revision, precision, prompt, generation config, batch и данным.
- Throughput, latency, RAM, VRAM, valid JSON rate, micro/macro и per-class F1.
- CSV с false positive/false negative и Streamlit demo с выбором extractor.
- Автоматические тесты загрузки, парсинга, оценки, cache и edge cases.

## Установка

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

INT8 через `bitsandbytes` рекомендуется запускать в Linux/WSL с NVIDIA CUDA. Веса обеих моделей должны быть доступны через Hugging Face или локальный cache.

## Подготовка реальных данных и gold-разметки

Сначала выгрузите 2 000 уникальных диалогов и создайте шаблон для независимой ручной разметки первых 200:

```powershell
python scripts/run_demo.py --sample-size 2000 --gold-size 200 --prepare-annotations
```

Заполните `data/gold_annotation_template.csv`: одна строка соответствует одной сущности или событию, допустимые labels — `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`. Сохраните проверенный файл как `data/gold_annotations.csv`. Итоговый запуск остановится, если размечено менее 200 различных диалогов.

## Итоговый benchmark

```powershell
python scripts/run_demo.py --sample-size 2000 --gold-size 200 --batch-sizes 1 2 4 8 --profiles rules spacy mistral-fp16 mistral-int8 openchat-fp16 openchat-int8 --rebuild-cache
```

Для отдельного профиля:

```powershell
python scripts/run_demo.py --profiles mistral-int8 --batch-sizes 1 2 4 8
```

`--allow-synthetic` и `--allow-incomplete-gold` предназначены только для разработки. Они не должны использоваться при формировании итоговой таблицы домашней работы.

## Smoke-тест без GPU

```powershell
python scripts/run_demo.py --sample-size 5 --gold-size 5 --batch-sizes 1 2 4 --profiles rules --allow-synthetic --rebuild-cache
python -m pytest tests -q
```

## Артефакты

- `data/benchmark_results.csv` — агрегированные метрики по профилям и batch size.
- `data/dataset_stats.json` — размер, уникальность и распределение источников выборки.
- `data/gold_stats.json` — объём gold-выборки и распределение классов.
- `data/per_class_metrics.csv` — precision/recall/F1 по каждому label.
- `data/extraction_errors.csv` — false positive/false negative.
- `data/extraction_predictions.csv` — извлечённые элементы и ошибки JSON.
- `data/extractions.json` — примеры полных структурированных ответов.
- `docs/pipeline.md` — устройство пайплайна.
- `docs/result_analyze.md` — проверенные результаты и правила итогового анализа.

## Demo UI

```powershell
streamlit run scripts/app.py
```

UI позволяет выбрать extractor, показывает время обработки, сущности, события, отношения, JSON и ошибку парсинга модели.
