# Анализ результатов

## Dataset filtering

Итоговая выборка формируется по цепочке: WildChat → rule-based high-recall prefilter → извлечение initial user intent → truncation classifier input → prompt-injection-resistant `Qwen/Qwen3-1.7B` classifier → `NEWS / NOT_NEWS`. Полный исходный conversation сохраняется в `NewsDialog.text` для IE benchmark, но Stage 2 получает только первое содержательное User-сообщение; после коротких `hi`, `hello`, `yes` или `continue` добавляется следующее User-сообщение. Assistant responses не входят в classifier input.

Ранняя версия классифицировала весь conversation. Длинный контекст замедлял inference, создавал false positive на постороннем содержимом Assistant и позволял инструкциям внутри WildChat влиять на classifier как prompt injection. Поэтому classifier input ограничен user intent, токенизируется с truncation и отделяется в prompt как недоверенные данные. Веса classifier скачиваются с Hugging Face Hub, а inference выполняется локально на CPU или CUDA.

Classifier используется только для подготовки датасета и не является benchmark-моделью. Основной эксперимент по-прежнему сравнивает локальные Mistral и OpenChat в FP16/INT8.

Первый вариант Stage 2 на `Qwen/Qwen2.5-0.5B-Instruct` оказался недостаточно надёжным: наблюдались false positive на troubleshooting и художественных сценариях, false negative на явных новостных запросах и большое количество `INVALID`. Поэтому он заменён на более сильную локальную `Qwen/Qwen3-1.7B` примерно на 2B параметров, а prompt стал явно отделять реальные события от fiction, hypothetical, historical, academic и troubleshooting-текстов. Фактическое качество новой модели должно подтверждаться встроенным sanity check и следующим реальным filtering run; итоговые числа до этого запуска не приводятся.

После реального запуска сюда следует перенести из `data/dataset_stats.json`:

- число просмотренных строк WildChat;
- число кандидатов после Stage 1;
- число принятых как `NEWS`;
- число отклонённых как `NOT_NEWS`;
- число `INVALID` и cache hits.

До повторного локального запуска classifier фактические значения не приводятся.

## Целевой эксперимент

Финальное сравнение выполняется на 200 реальных новостных диалогах WildChat и 10 вручную размеченных gold-диалогах:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 10 `
  --batch-sizes 1 2 4 8 `
  --profiles mistral-fp16 mistral-int8 openchat-fp16 openchat-int8 `
  --rebuild-cache
```

Для каждой модели сравниваются FP16 и INT8 по throughput, latency, RAM, VRAM, precision, recall и F1. Затем Mistral и OpenChat сравниваются при одинаковом precision mode и batch size.

## Статус результатов

Фактический финальный GPU benchmark ещё должен быть выполнен после пересоздания выборки новым фильтром и завершения ручной разметки. Существующие smoke-артефакты нельзя использовать как итоговое сравнение моделей.

Rules и spaCy являются только optional baselines. Они могут показать разницу между быстрым специализированным подходом и более дорогим LLM inference, но не входят в обязательные четыре профиля.

## Интерпретация

- Throughput оценивать вместе с p95 latency и максимально устойчивым batch size.
- VRAM сравнивать между FP16 и INT8 одной модели.
- Основными показателями качества считать micro precision, recall и F1.
- Macro/per-class метрики использовать осторожно: 10 gold-диалогов могут слабо покрывать редкие labels.
- Ошибки разбирать по неоднозначным PERSON/ORG, нескольким событиям, датам без года, отсутствующему SOURCE и связям EVENT–IMPACT.

Ожидается, что INT8 уменьшит потребление VRAM и позволит увеличить batch size, но фактическая скорость и изменение качества должны подтверждаться сохранёнными результатами, а не предположениями.
