# Pipeline

## Данные

`allenai/WildChat-1M` читается в streaming-режиме. Запись принимается, если она англоязычная и содержит новостные маркеры. Дубликаты удаляются одновременно по `dialog_id` и SHA-256 нормализованного текста. Итоговый benchmark требует 2 000 уникальных записей без источника `synthetic`.

Синтетические примеры используются только с `--allow-synthetic` для тестирования. При недоступности WildChat финальный режим завершается ошибкой и не подменяет реальные данные fallback-набором.

## Gold-разметка

Команда с `--prepare-annotations` создаёт CSV-шаблон для первых 200 реальных диалогов. Gold-файл заполняется независимо вручную; предсказания rules или LLM не копируются в эталон. Перед benchmark проверяются labels и количество размеченных диалогов.

Качество считается по строгому совпадению `(dialog_id, label, normalized value)`. Сохраняются micro F1, macro F1, per-class precision/recall/F1, support, false positive и false negative.

## Извлечение

Профили:

| Профиль | Модель | Precision |
| --- | --- | --- |
| `rules` | регулярные выражения и словари | native |
| `spacy` | `en_core_web_sm` + rules | native |
| `mistral-fp16` | `mistralai/Mistral-7B-Instruct-v0.2` | FP16 |
| `mistral-int8` | та же модель | INT8 |
| `openchat-fp16` | `openchat/openchat-3.5-0106` | FP16 |
| `openchat-int8` | та же модель | INT8 |

LLM получает prompt через собственный chat template. Генератор возвращает только продолжение без исходного prompt. Парсер ищет JSON с массивами `entities`, `events`, `relations`, отбрасывает неизвестные labels и сохраняет `parse_valid` и текст ошибки.

## Benchmark и cache

Каждый профиль проверяется с batch size `1/2/4/8`. CUDA OOM помечает размер как неуспешный, очищает GPU cache и не прерывает остальные размеры. Максимальный успешный batch выводится в консоль.

Измеряются время загрузки, docs/sec, chars/sec, оценка tokens/sec, mean/p95 latency, peak process RAM, peak allocated CUDA VRAM и valid JSON rate. Fingerprint cache учитывает модель, revision, precision, версию prompt, generation config, batch size, идентификаторы и хэши текстов. Рядом с extraction cache хранится исходный snapshot метрик, поэтому cache-hit не создаёт нулевые значения скорости.

## Выходные данные

Один CLI-запуск формирует общую таблицу всех профилей и batch size, per-class метрики, ошибки и предсказания. Синтетический smoke benchmark не считается итоговым результатом домашней работы.
