# Pipeline

## Требования

- Python 3.
- Hugging Face токен с доступом к Inference Providers.
- Чат-модель `Qwen/Qwen3.5-9B`.
- Embedding-модель `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Проект Langfuse и его ключи — если нужны traces и LLM-оценка в Langfuse Cloud.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Настройка

Создаём локальный `.env` из примера:

```powershell
Copy-Item .env.example .env
```

Hugging Face токен создаётся в [User Access Tokens](https://huggingface.co/settings/tokens): нажмите `New token`, выберите `Read` или fine-grained токен с доступом к Inference Providers и скопируйте значение `hf_...`.

Langfuse ключи создаются в проекте Langfuse: `Settings` → `API Keys`. Нужны `Public key`, `Secret key` и host проекта.

Заполненный локальный `.env`:

```env
HF_TOKEN=hf_...
HF_PROVIDER=auto
HF_CHAT_MODEL=Qwen/Qwen3.5-9B
HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Параметры Langfuse необязательны для RAG: без ключей приложение работает, но не отправляет traces, scores и events в Langfuse.

## Запуск

```powershell
cd D:\LLMs\LLM_course_homeworks\12_files_guide

.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m streamlit run scripts/app.py
```

Открываем `http://localhost:8501`.

## Работа приложения

1. Добавляем в боковой панели файлы `.md` или `.txt` либо папку с такими файлами.
2. Нажимаем «Сохранить и обновить индекс». Документы сохраняются в `data/documents`, разбиваются на фрагменты и индексируются в локальной Chroma.
3. Задаём вопрос. Приложение находит до четырёх релевантных фрагментов, передаёт их вместе с вопросом модели `Qwen/Qwen3.5-9B` и показывает ответ с источниками.
4. Нажимаем `Clear files`, чтобы удалить документы и локальные коллекции Chroma.

```text
Markdown и text-файлы
        ↓
разбиение на фрагменты
        ↓
embedding-модель Hugging Face
        ↓
локальная Chroma
        ↓
поиск до 4 релевантных фрагментов
        ↓
вопрос + контекст → Qwen/Qwen3.5-9B
        ↓
ответ и источники в Streamlit
```

## Langfuse

При настроенных ключах каждый пользовательский вопрос записывается в Langfuse как trace `answer_question`. Внутри trace создаются observation `retrieve_context` и generation `generate_answer`; в generation сохраняются входные и выходные данные, задержка и usage модели. Также приложение создаёт events для загрузки документов, обновления индекса, очистки файлов и ошибок.

Кнопка «Создать датасет Langfuse» один раз создаёт датасет `knowledge_base_evaluation` с двумя тестовыми вопросами и критериями ожидаемых ответов. Повторное нажатие не создаёт дубликат. Обычные вопросы и обновление индекса не изменяют датасет.

Кнопка «Запустить LLM-оценку» создаёт Langfuse Experiment на датасете `knowledge_base_evaluation`. RAG формирует ответ для каждого тестового вопроса, затем `Qwen/Qwen3.5-9B` оценивает его и записывает score `answer_groundedness` со значением `0` или `1`. Experiment выполняется последовательно.

Для generation в Langfuse передаются usage токенов из ответа Hugging Face. Стоимость указывается как `0`, если provider не возвращает стоимость в ответе.
