# Анализ результатов

## Dataset filtering

Итоговая выборка формируется в два этапа: rule-based high-recall prefilter сокращает поток WildChat, после чего отдельный LLM classifier принимает финальное решение `NEWS / NOT_NEWS`. Это понадобилось из-за false positive чистого rule-based подхода: художественных текстов, chapter review, code/programming, jailbreak и advertising prompts.

Classifier используется только для подготовки датасета и не является benchmark-моделью. Основной эксперимент по-прежнему сравнивает локальные Mistral и OpenChat в FP16/INT8.

После реального запуска сюда следует перенести из `data/dataset_stats.json`:

- число просмотренных строк WildChat;
- число кандидатов после Stage 1;
- число принятых как `NEWS`;
- число отклонённых как `NOT_NEWS`;
- число `INVALID` и cache hits.

До повторного запуска с настроенным classifier фактические значения не приводятся.

## Целевой эксперимент

Финальное сравнение выполняется на 200 реальных новостных диалогах WildChat и 20 вручную размеченных gold-диалогах:

```powershell
python scripts/run_demo.py `
  --sample-size 200 `
  --gold-size 20 `
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
- Macro/per-class метрики использовать осторожно: 20 gold-диалогов могут слабо покрывать редкие labels.
- Ошибки разбирать по неоднозначным PERSON/ORG, нескольким событиям, датам без года, отсутствующему SOURCE и связям EVENT–IMPACT.

Ожидается, что INT8 уменьшит потребление VRAM и позволит увеличить batch size, но фактическая скорость и изменение качества должны подтверждаться сохранёнными результатами, а не предположениями.
