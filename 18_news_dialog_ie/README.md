# NER / IE для новостных диалогов

Выбран **трек B — новостные диалоги**.

## Задача

Нужно:

- получить новостные диалоги из WildChat;
- извлечь сущности `PERSON`, `ORG`, `LOC`, `EVENT`, `DATE`, `IMPACT`, `SOURCE`;
- сравнить несколько локальных моделей;
- сравнить FP16 и INT8;
- измерить качество и производительность.

---

## Датасет

Используется:

```text
allenai/WildChat-1M
```

WildChat содержит много диалогов не по новостной теме, поэтому используется двухступенчатый отбор:

```text
WildChat
→ rule-based prefilter
→ Qwen3-1.7B NEWS / NOT_NEWS classifier
→ 200 новостных диалогов
```

Classifier запускается локально через Hugging Face Transformers.

Для классификации используется только первое содержательное сообщение пользователя, без ответов Assistant. Вход ограничен 1024 токенами.

Перед обработкой выполняется sanity check classifier-а.

---

## Gold-разметка

Из 200 диалогов выбираются:

```text
10 gold dialogs
```

Они размечаются вручную через Streamlit:

```bash
streamlit run scripts/annotation_app.py
```

Gold используется для расчёта Precision / Recall / F1.

---

## Сравниваемые подходы

Baseline:

```text
Rules
spaCy
```

Основные LLM:

```text
Mistral FP16
Mistral INT8

OpenChat FP16
OpenChat INT8
```

Основная цель — сравнить:

```text
качество
скорость
RAM / VRAM
```

---

## Batch processing

Проверяются batch sizes:

```text
1
2
4
8
```

Для каждого профиля измеряются:

- throughput;
- latency;
- tokens/sec;
- RAM;
- VRAM.

---

## Метрики качества

На Gold subset считаются:

- Precision;
- Recall;
- F1;
- Micro F1;
- Macro F1;
- Per-class F1.

Также анализируются:

- False Positive;
- False Negative.

---

## Общий pipeline

```text
WildChat-1M
    ↓
Rule-based prefilter
    ↓
Qwen3-1.7B NEWS classifier
    ↓
200 news dialogs
    ↓
10 manual gold dialogs
    ↓
NER / IE
    ├── Rules
    ├── spaCy
    ├── Mistral FP16
    ├── Mistral INT8
    ├── OpenChat FP16
    └── OpenChat INT8
    ↓
Quality + performance benchmark
```

---

## Подготовка данных

```powershell
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

---

## Что проверяется

- Mistral vs OpenChat;
- FP16 vs INT8;
- влияние batch size;
- качество NER / IE;
- расход RAM / VRAM;
- trade-off между скоростью и качеством.

Фактические результаты эксперимента сохраняются в:

```text
docs/result_analyze.md
```

### Ограничение VRAM для Mistral

На RTX 2060 SUPER 8 GB профиль Mistral FP16 потребовал CPU/disk offload и завершился native crash. Поэтому его результаты не приводятся как завершённый benchmark.

Профиль Mistral INT8 использует heterogeneous inference с CPU offload из-за ограничения VRAM. Основная часть linear weights квантизована в INT8, а модули, выгруженные на CPU, остаются в FP32. Профиль при этом учитывается как `mistral-int8`; throughput и F1 следует указывать только после реального успешного запуска.
