# Pipeline

## Требования

- Python 3.
- Запущенный Ollama.
- Модели `qwen2.5:3b` и `bge-m3`.
- Проект Langfuse и его ключи — если нужны traces и LLM-оценка в Langfuse Cloud.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull bge-m3
```

## Настройка

Локальный `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:3b
OLLAMA_EMBEDDING_MODEL=bge-m3
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Параметры Langfuse необязательны для RAG: без ключей приложение работает локально, но не отправляет traces, scores и events в Langfuse.

## Запуск

```powershell
cd D:\LLMs\LLM_course_homeworks\12_Frameworks-and-agents

.\.venv\Scripts\Activate.ps1
streamlit run scripts/app.py
```

Открываем `http://localhost:8501`.

## Работа приложения

1. Добавляем в боковой панели файлы `.md` или `.txt` либо папку с такими файлами.
2. Нажимаем «Сохранить и обновить индекс». Документы сохраняются в `data/documents`, разбиваются на фрагменты и индексируются в локальной Chroma.
3. Задаём вопрос. Приложение находит до четырёх релевантных фрагментов, передаёт их вместе с вопросом модели `qwen2.5:3b` и показывает ответ с источниками.
4. Нажимаем `Clear files`, чтобы удалить документы и локальные коллекции Chroma.

```text
Markdown и text-файлы
        ↓
разбиение на фрагменты
        ↓
embedding-модель bge-m3
        ↓
локальная Chroma
        ↓
поиск до 4 релевантных фрагментов
        ↓
вопрос + контекст → qwen2.5:3b
        ↓
ответ и источники в Streamlit
```

## Langfuse

При настроенных ключах каждый пользовательский вопрос записывается в Langfuse как trace `answer_question`. Внутри trace создаются observation `retrieve_context` и generation `generate_answer`; в generation сохраняются входные и выходные данные, задержка и usage модели. Также приложение создаёт events для загрузки документов, обновления индекса, очистки файлов и ошибок.

Кнопка «Создать датасет Langfuse» один раз создаёт датасет `knowledge_base_evaluation` с двумя тестовыми вопросами и критериями ожидаемых ответов. Повторное нажатие не создаёт дубликат. Обычные вопросы и обновление индекса не изменяют датасет.

Кнопка «Запустить LLM-оценку» создаёт Langfuse Experiment на датасете `knowledge_base_evaluation`. RAG формирует ответ для каждого тестового вопроса, затем локальная `qwen2.5:3b` оценивает его и записывает score `answer_groundedness` со значением `0` или `1`. Experiment выполняется последовательно, так как обе роли использует одна локальная модель.

Локальная Ollama не тарифицируется, поэтому для generation явно передаётся нулевая стоимость. Langfuse сохраняет стоимость как `0`, а токены — по данным ответа Ollama.
