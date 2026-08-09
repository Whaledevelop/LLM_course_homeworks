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
которые подходят под доступную VRAM. Эксперимент с 7B показывает влияние
аппаратных ограничений и offload на производительность.

## FP16 vs INT8 --- промежуточный результат

Smoke benchmark Qwen3-1.7B при `batch=8`:

  Precision        Throughput        Latency      VRAM
  ----------- --------------- -------------- ---------
  FP16          0.41 docs/sec   2.42 sec/doc   5.56 GB
  INT8          0.11 docs/sec   8.77 sec/doc   4.47 GB

Вывод:

-   FP16 полностью помещается в VRAM без CPU offload.
-   INT8 экономит примерно **1.1 GB VRAM**.
-   При этом INT8 примерно в **3.7 раза медленнее FP16** по throughput.
-   В данной конфигурации quantization даёт выигрыш по памяти, но не по
    скорости.
-   Для Qwen3-1.7B на RTX 2060 SUPER FP16 оказался более эффективным
    режимом.
-   Увеличение batch size заметно повышает throughput FP16: при
    `batch=8` достигнуто `0.41 docs/sec`.

Это smoke-результаты на 8 диалогах. Финальные метрики фиксируются после
полного benchmark.

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
  --batch-sizes 1 2 4 8 `
  --profiles qwen-fp16 qwen-int8 gemma-fp16 gemma-int8 `
  --rebuild-cache
```

Подробные результаты сохраняются в `docs/result_analyze.md`.
