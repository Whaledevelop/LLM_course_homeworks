# RAG-система с поиском по собственной базе документов

Домашнее задание реализовано как локальный Python-проект с ChromaDB. Проект создает синтетический датасет на 1200 документов по темам векторных БД, ANN-алгоритмов, RAG, фильтрации, оценки качества и hybrid search, строит эмбеддинги, индексирует документы и запускает end-to-end RAG-поиск.

## Что сделано

- Выбрана векторная БД ChromaDB.
- Настроена persistent-коллекция с HNSW-индексом.
- Создан датасет 1000+ документов с метаданными `category`, `source`, `year`, `difficulty`.
- Реализованы локальные эмбеддинги без внешних API-ключей.
- Реализован семантический поиск по векторам.
- Настроены similarity metrics: `cosine`, `l2`, `ip`.
- Реализована фильтрация по метаданным.
- Реализована настройка `top-k`.
- Реализован hybrid search: vector search + keyword scoring + Reciprocal Rank Fusion.
- Добавлены замеры Recall@K, MRR и latency.
- Добавлен RAG-ответ на основе найденных источников.
- Добавлен batch benchmark для нескольких HNSW-конфигураций.

## Установка

Требуется Python 3.10-3.13. Не используйте Python 3.14: часть зависимостей ChromaDB ставит бинарные wheel-файлы, и при смешивании версий появляется ошибка `numpy._core._multiarray_umath`.

Если окружение уже создано под Python 3.14 или зависимости установлены с ошибкой, пересоздайте его:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
.\setup.ps1
.\.venv\Scripts\activate
```

Альтернативно можно создать окружение вручную через установленный Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Запуск

```bash
python scripts/run_demo.py --rebuild
```

Запуск с полным сравнением ANN-настроек:

```bash
python scripts/run_demo.py --rebuild --benchmark
```

Запуск со своим запросом:

```bash
python scripts/run_demo.py --query "How does metadata filtering improve RAG retrieval?"
```

## Структура

- `scripts/dataset.py` - генерация и загрузка датасета.
- `scripts/embedding_service.py` - локальная embedding-функция.
- `scripts/index_service.py` - настройка ChromaDB, HNSW-параметры и индексация.
- `scripts/search_service.py` - semantic search, filtered search и hybrid search.
- `scripts/rag_service.py` - end-to-end RAG-ответ с источниками.
- `scripts/evaluation.py` - Recall@K, MRR, latency и benchmark профилей.
- `scripts/run_demo.py` - воспроизводимый сценарий запуска.
- `docs/pipeline.md` - описание пайплайна и конфигурации.
- `docs/result_analyze.md` - анализ результатов и trade-offs.
- `data/documents.jsonl` - генерируется при запуске.
- `data/chroma` - локальная persistent-база ChromaDB, генерируется при запуске.
- `data/benchmark_results.csv` - генерируется при запуске с `--benchmark`.

## Вывод

ChromaDB подходит для учебного и прототипного RAG: ее можно поднять без отдельного сервера, быстро создать коллекцию, хранить документы вместе с метаданными и тестировать HNSW-параметры. Для production-сценариев с жесткими требованиями к масштабированию, шардированию и управлению нагрузкой стоит сравнить Chroma с Qdrant, Milvus или Pinecone.
