# NER / IE для новостных диалогов

Выбран **трек B --- новостные диалоги**.

## Задача

-   получить новостные диалоги из WildChat;
-   извлечь `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`;
-   сравнить локальные модели и FP16 / INT8;
-   измерить качество и производительность.

## Датасет

Используется `allenai/WildChat-1M`.

Двухступенчатый отбор:

``` text
WildChat
→ rule-based prefilter
→ Qwen3-1.7B NEWS / NOT_NEWS classifier
→ 200 новостных диалогов
```

Классификатор запускается локально через Hugging Face Transformers.
Используется первое содержательное сообщение пользователя, вход
ограничен 1024 токенами.

## Gold-разметка

Из 200 диалогов выбираются 10 gold dialogs. Разметка выполняется через:

``` bash
streamlit run scripts/annotation_app.py
```

Gold используется для расчёта Precision / Recall / F1.

## Benchmark subset

Исходный `data/news_dialogs.jsonl` содержит 200 NEWS dialogs и не сокращается. Из-за времени локального inference на RTX 2060 SUPER 8 GB основной inference benchmark по умолчанию использует 50 диалогов: все 10 gold и 40 самых коротких non-gold.

- `--benchmark-size` задаёт общий размер inference subset, default — `50`;
- `--benchmark-selection shortest` добирает самые короткие non-gold по числу символов;
- `--benchmark-selection first` добирает первые non-gold в порядке dataset;
- deprecated `--benchmark-limit` сохраняется как alias для `--benchmark-size`.

Выбранный состав сохраняется в `data/benchmark_subset.jsonl`. Ни одна стратегия не фильтрует gold по длине и не пересоздаёт WildChat dataset.

## Сравниваемые подходы

Baseline: - Rules - spaCy

LLM: - Qwen3-1.7B FP16 - Qwen3-1.7B INT8 - Gemma 2 2B IT FP16 - Gemma 2
2B IT INT8

Проверяются batch sizes `1 / 2 / 4 / 8`, throughput, latency,
tokens/sec, RAM и VRAM.

## Pipeline

``` text
WildChat-1M
    ↓
Rule-based prefilter
    ↓
Qwen3-1.7B NEWS classifier
    ↓
200 news dialogs
    ↓
10 gold dialogs
    ↓
50-dialog benchmark subset
    ↓
NER / IE
    ├── Rules
    ├── spaCy
    ├── Qwen3-1.7B FP16 / INT8
    └── Gemma 2 2B IT FP16 / INT8
    ↓
Quality + performance benchmark
```

## Ограничения 7B-моделей

На RTX 2060 SUPER 8 GB Mistral-7B не помещается полностью в VRAM.

FP16 потребовал сильного CPU/disk offload и завершился native crash.
Mistral INT8 загрузился с CPU offload последних слоёв, но inference
первого документа при `batch=1` занял около 165.9 сек.

Поэтому основной эксперимент перенесён на Qwen3-1.7B и Gemma 2 2B IT,
которые подходят под доступную VRAM.

Эксперимент с 7B показывает влияние аппаратных ограничений и offload на
производительность.

## Результаты benchmark

Smoke benchmark на 8 диалогах, `batch_size=8`:

| Модель | Precision | Throughput | Latency | RAM | VRAM | Micro F1 | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | FP16 | **0.41 docs/sec** | **2.42 sec/doc** | 1371 MB | 5557 MB | 0.000 | 0.000 |
| Qwen3-1.7B | INT8 | 0.11 docs/sec | 8.77 sec/doc | 1444 MB | **4472 MB** | 0.000 | 0.000 |
| Gemma 2 2B IT | FP16 | 0.09 docs/sec | 11.33 sec/doc | 2205 MB | 7440 MB | **0.018** | **0.024** |
### FP16 vs INT8

Для Qwen3-1.7B:

-   FP16 полностью помещается в VRAM без CPU offload.
-   INT8 экономит примерно **1.1 GB VRAM**.
-   INT8 примерно в **3.7 раза медленнее FP16** по throughput.
-   В данной конфигурации quantization даёт выигрыш по памяти, но не по
    скорости.
-   Для RTX 2060 SUPER FP16 оказался более эффективным режимом.

### Влияние batching

Для Qwen3-1.7B FP16:

### Влияние batch size на Qwen3-1.7B FP16

| Batch size | Throughput | Latency |
| ---: | ---: | ---: |
| 1 | 0.05 docs/sec | 18.55 sec/doc |
| 2 | 0.09 docs/sec | 11.25 sec/doc |
| 4 | 0.13 docs/sec | 7.52 sec/doc |
| 8 | **0.41 docs/sec** | **2.42 sec/doc** |

Увеличение batch size заметно повышает throughput, поскольку несколько
диалогов обрабатываются моделью параллельно на GPU.

### Qwen vs Gemma

-   Qwen3-1.7B FP16 --- самый быстрый протестированный вариант.
-   Gemma 2 2B FP16 полностью помещается в GPU без CPU offload, но
    использует около **7.44 GB VRAM**.
-   Gemma существенно медленнее Qwen FP16.
-   Gemma показала ненулевой F1, в отличие от текущих результатов Qwen.
-   Низкий F1 требует отдельной проверки промпта, парсинга результатов и
    gold-разметки перед финальным benchmark.

Текущие результаты являются smoke benchmark и используются для проверки
работоспособности моделей и выбора конфигурации перед полным прогоном.

### Quality debug

Флаг `--quality-debug` печатает для каждого gold-диалога raw response модели, результат JSON parsing, извлечённые entities/events/relations и нормализованные `gold`, `predicted`, `true positives`, `false positives`, `false negatives`. Режим не меняет prompt, extraction или evaluation и работает также с результатами из cache.

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --benchmark-size 5 `
  --benchmark-selection first `
  --batch-sizes 1 `
  --profiles qwen-fp16 `
  --quality-debug
```

## Подготовка данных

``` powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --prepare-annotations `
  --rebuild-dataset `
  --rebuild-classifier-cache `
  --news-classifier-model Qwen/Qwen3-1.7B `
  --news-classifier-batch-size 4 `
  --news-classifier-max-input-tokens 1024 `
  --news-classifier-sanity-check
```

## Полный benchmark

``` powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --benchmark-size 50 `
  --benchmark-selection shortest `
  --batch-sizes 4 `
  --profiles qwen-fp16 `
  --rebuild-cache
```

Подробные результаты сохраняются в `docs/result_analyze.md`.
