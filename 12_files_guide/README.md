# Справочник по загруженным документам

RAG-приложение на Streamlit. Пользователь загружает Markdown и text-файлы, после чего может задавать вопросы только по их содержимому. Для генерации используется `Qwen/Qwen3.5-9B` через Hugging Face Inference Providers, для embedding — Hugging Face feature extraction, для наблюдаемости — Langfuse.

## Документация

- [Pipeline](docs/pipeline.md) — установка, настройка, запуск и схема работы приложения.
- [Анализ результатов](docs/result_analyze.md) — методика LLM-оценки и работа с результатами в Langfuse.

## Основные пути

### `./docs`

Документация по запуску приложения и анализу его оценки.

### `./data`

`documents` — документы, добавленные через интерфейс. `chroma` — локальные коллекции векторной базы Chroma. Обе папки являются рабочими данными приложения и исключены из Git.

### `./scripts/app.py`

Streamlit-интерфейс: загрузка и очистка документов, обновление индекса, чат, создание датасета Langfuse и запуск LLM-оценки.

### `./scripts/rag_service.py`

Загрузка документов, разбиение на фрагменты, создание индекса Chroma, поиск контекста и вызов Hugging Face моделей.

### `./scripts/evaluation_service.py`

Создание датасета `knowledge_base_evaluation` и оценка ответов локальной моделью-судьёй.

### `./scripts/observability.py`

Подключение к Langfuse и запись traces, observations, scores и событий приложения.

### `./scripts/settings.py`

Чтение параметров из `.env` и хранение путей, URL и названий моделей.

### `./requirements.txt`

Python-зависимости проекта.

### `./.env`

Локальная конфигурация Hugging Face и ключи Langfuse. Файл не должен попадать в Git.
