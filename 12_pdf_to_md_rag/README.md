# PDF RAG: PDF → Markdown

Локальное RAG-приложение на Streamlit.

Возможности:

* загрузка одного или нескольких PDF;
* конвертация PDF в Markdown;
* извлечение встроенных изображений;
* сохранение изображений на диск;
* построение Chroma-индекса по Markdown;
* вопросы к документам через Ollama;
* просмотр найденных источников.

## Требования

* Python 3.11+
* Ollama

## Установка

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать окружение:

### Windows

```bash
.venv\\Scripts\\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Создать `.env`:

```bash
cp .env.example .env
```

## Ollama

Скачать модели:

```bash
ollama pull qwen2.5:3b
ollama pull bge-m3
```

Запустить Ollama:

```bash
ollama serve
```

## Запуск

```bash
streamlit run scripts/app.py
```

После запуска открыть адрес:

```text
http://localhost:8501
```

## Структура данных

```text
data/
├── documents/
│   ├── my_file.pdf
│   ├── my_file.md
│   └── my_file_assets/
│       └── images/
└── chroma/
```

## Pipeline

```text
PDF
↓
Markdown + Images
↓
Chunking
↓
Embeddings
↓
Chroma
↓
Retriever
↓
Ollama
↓
Answer
```

## Ограничения

* OCR не реализован.
* Индексируются только Markdown-файлы.
* Ответы строятся только на основе найденного контекста.
* Качество конвертации зависит от структуры PDF.
* Сложная вёрстка PDF может преобразовываться неточно.
