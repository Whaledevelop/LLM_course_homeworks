# NER / IE для новостных диалогов

Домашняя работа по **треку B: извлечение сущностей и событий из новостных диалогов**.

## Что реализовано

- Датасет: `allenai/WildChat-1M`.
- Извлекаемые типы: `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`.
- Локальный inference через Hugging Face Transformers.
- Сравнение **Qwen3-1.7B** и **Gemma 2 2B IT**.
- Сравнение **FP16 / INT8**.
- Batch processing, throughput, latency, RAM/VRAM.
- Оценка качества по вручную размеченному gold-набору.

## Подготовка данных

```text
WildChat-1M
→ rule-based prefilter
→ локальный Qwen3-1.7B NEWS / NOT_NEWS classifier
→ 200 NEWS dialogs
```

Для оценки качества вручную размечены **10 gold dialogs**.

Основной benchmark выполняется на **50 диалогах**: все 10 gold и 40 самых коротких non-gold. Исходный набор из 200 диалогов сохраняется.

Небольшой размер gold-набора и benchmark subset выбран из-за аппаратных ограничений тестовой машины: **RTX 2060 SUPER 8 GB**. На более крупных выборках локальный inference занимал слишком много времени, а для 7B-моделей требовался CPU/disk offload. Поэтому для практического завершения домашней работы использованы 10 gold и 50 benchmark dialogs при сохранении полного подготовленного набора из 200 диалогов.

## Структура проекта и результаты

| Путь | Назначение |
| --- | --- |
| `scripts/` | Подготовка данных, разметка и запуск benchmark. Основной скрипт — `scripts/run_demo.py`. |
| `data/` | Датасет, gold-разметка, benchmark subset, кэш и результаты запусков. |
| `docs/` | Дополнительные материалы и подробный анализ результатов. |

### Где смотреть результаты

| Файл | Содержимое |
| --- | --- |
| `data/benchmark_results.csv` | Численные результаты: модель, precision, batch size, throughput, latency, RAM/VRAM и F1. |
| `data/benchmark_subset.jsonl` | 50 диалогов, использованных в финальном benchmark. |
| `data/benchmark_failures.csv` | Неуспешные запуски и причины ошибок. |
| `data/news_dialogs.jsonl` | Полный подготовленный набор из 200 новостных диалогов. |
| `docs/result_analyze.md` | Подробный текстовый анализ экспериментов. |

Главный результат домашней работы — таблица **«Финальный benchmark»** ниже: сравнение четырёх конфигураций по скорости, памяти и качеству. Лучший итоговый вариант — **Qwen3-1.7B FP16**.

## Локальные модели

Изначально тестировался `Mistral-7B-Instruct-v0.2`, но модель не помещалась полностью в 8 GB VRAM.

- FP16 потребовал CPU/disk offload и завершался native crash.
- INT8 использовал CPU offload; inference одного документа занимал около **165.9 сек**.

Поэтому основной benchmark проведён на моделях, полностью помещающихся в GPU без CPU offload:

- `Qwen/Qwen3-1.7B`
- `google/gemma-2-2b-it`

## Финальный benchmark

Одинаковые условия: **50 dialogs, 10 gold, batch=4, selection=shortest**.

| Модель | Precision | Throughput | Latency | RAM | VRAM | Micro F1 | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | FP16 | **0.14 docs/sec** | **7.36 sec/doc** | **1368 MB** | 5198 MB | **0.357** | **0.322** |
| Qwen3-1.7B | INT8 | 0.04 docs/sec | 24.36 sec/doc | 1463 MB | **4054 MB** | 0.279 | 0.265 |
| Gemma 2 2B IT | FP16 | 0.09 docs/sec | 11.44 sec/doc | 1596 MB | 7148 MB | 0.233 | 0.210 |
| Gemma 2 2B IT | INT8 | 0.02 docs/sec | 40.40 sec/doc | 1533 MB | 5460 MB | 0.236 | 0.223 |

### Выводы

- **Qwen3-1.7B FP16** показал лучшие скорость и качество.
- Qwen INT8 экономит около **1.14 GB VRAM**, но throughput падает примерно в **3.5 раза**.
- Gemma INT8 экономит около **1.69 GB VRAM**, но также существенно медленнее FP16.
- Gemma 2 2B требует больше VRAM и уступает Qwen3-1.7B по скорости и качеству.
- Для RTX 2060 SUPER 8 GB оптимальной конфигурацией оказалась **Qwen3-1.7B FP16**.

## Влияние batching

Smoke-тест Qwen3-1.7B FP16:

| Batch size | Throughput | Latency |
| ---: | ---: | ---: |
| 1 | 0.05 docs/sec | 18.55 sec/doc |
| 2 | 0.09 docs/sec | 11.25 sec/doc |
| 4 | 0.13 docs/sec | 7.52 sec/doc |
| 8 | 0.41 docs/sec | 2.42 sec/doc |

Увеличение batch size повышает throughput. `batch=8` успешно работал на коротком smoke-тесте, но при полном прогоне вызвал CUDA OOM, поэтому для финального benchmark выбран стабильный `batch=4`.

## Запуск финального benchmark

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --benchmark-size 50 `
  --benchmark-selection shortest `
  --batch-sizes 4 `
  --profiles qwen-fp16
```

Результаты сохраняются в `data/benchmark_results.csv`, состав subset — в `data/benchmark_subset.jsonl`.
