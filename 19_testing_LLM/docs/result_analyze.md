# Анализ результатов

Проверочный сценарий:

```powershell
python scripts/run_demo.py --rebuild-data
python -m pytest -q
```

Фактический результат локального запуска:

| Метрика | Значение | Порог | Статус |
| --- | ---: | ---: | --- |
| Faithfulness | 1.000 | 0.700 | pass |
| Answer relevance | 0.847 | 0.650 | pass |
| Context recall | 1.000 | 0.750 | pass |
| Pass rate по отдельным примерам | 0.750 | informational | не используется как hard gate |

`pytest` результат:

```text
1 passed in 0.04s
```

Demo run сформировал:

- `data/corpus.jsonl`;
- `data/ragas_results.json`;
- `data/ragas_results.html`.

## Метрики

Используются три Ragas-метрики из задания:

| Метрика | Что проверяет | Порог |
| --- | --- | ---: |
| Faithfulness | Ответ опирается на найденный контекст и не добавляет неподдержанные факты | 0.70 |
| Answer relevance | Ответ отвечает на вопрос и совпадает с expected facts/reference | 0.65 |
| Context recall | Retriever нашел документы, нужные для эталонного ответа | 0.75 |

## Trade-offs

Локальный evaluator не заменяет полноценный Ragas + LLM-as-a-judge на сложных открытых ответах. Зато он:

- не требует API-ключей;
- детерминированно работает в CI;
- быстро показывает регрессию retriever/generation;
- сохраняет ту же структуру данных, что и Ragas evaluation.

Полноценный production-вариант можно расширить внешним judge LLM, ручной валидацией части выборки, canary prompts и мониторингом drift после релиза.

## Вывод

Домашняя работа закрывает обязательную часть: есть LLM/RAG-приложение, goldens, проверка hallucination/faithfulness, relevance и context recall, pytest quality gates, CI/CD workflow и отчеты в JSON/HTML.
