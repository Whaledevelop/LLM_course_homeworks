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
Qwen3-1.7B FP16
Qwen3-1.7B INT8

Gemma 2 2B IT FP16
Gemma 2 2B IT INT8
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
    ├── Qwen3-1.7B FP16
    ├── Qwen3-1.7B INT8
    ├── Gemma 2 2B IT FP16
    └── Gemma 2 2B IT INT8
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

- Qwen3-1.7B vs Gemma 2 2B IT;
- FP16 vs INT8;
- влияние batch size;
- качество NER / IE;
- расход RAM / VRAM;
- trade-off между скоростью и качеством.

Фактические результаты эксперимента сохраняются в:

```text
docs/result_analyze.md
```

### Ограничения 7B-моделей

На RTX 2060 SUPER 8 GB модели Mistral-7B в FP16 и INT8 не помещаются полностью в VRAM. FP16 требует сильного CPU/disk offload и оказался нестабилен: запуск завершился native crash.

Mistral INT8 загрузился с CPU offload последних слоёв; фактически наблюдавшийся inference первого документа при `batch=1` занял около 165.9 секунды. Такой heterogeneous режим непрактичен для benchmark на 200 диалогах. Поэтому основной эксперимент использует локальные instruction-модели `Qwen/Qwen3-1.7B` (1.7B параметров) и `google/gemma-2-2b-it` (2B), которые должны полностью или почти полностью помещаться в VRAM. Эксперимент с 7B сохраняется как демонстрация trade-off между размером модели, precision и аппаратными ограничениями.

Gemma 2 является gated-моделью на Hugging Face: перед первым запуском нужно принять лицензию Google и авторизоваться.

### Smoke benchmark

`--benchmark-limit N` запускает benchmark только на первых N диалогах уже готового 200-dialog dataset и не пересоздаёт его. Метрики качества считаются только для gold-диалогов, попавших в subset.

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --benchmark-limit 5 `
  --batch-sizes 1 `
  --profiles qwen-fp16 qwen-int8 gemma-fp16 gemma-int8
```

Полный benchmark:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --batch-sizes 1 2 4 8 `
  --profiles qwen-fp16 qwen-int8 gemma-fp16 gemma-int8 `
  --rebuild-cache
```
