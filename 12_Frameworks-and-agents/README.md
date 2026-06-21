# RAG-тьютор по уроку 12

Первый этап домашней работы: локальный RAG без Langfuse. База знаний проекта — `src/lesson_12_frameworks_and_agents.md`.

## Как устроен запрос

```text
Markdown-конспект
-> разбиение на фрагменты
-> Ollama embedding model bge-m3
-> локальная Chroma

Вопрос из браузера
-> поиск 4 похожих фрагментов в Chroma
-> вопрос + найденный контекст
-> Ollama chat model qwen2.5:3b
-> ответ и источники в Streamlit
```

Ollama запускает модели как локальный HTTP-сервис. `bge-m3` превращает фрагменты и вопрос в векторы для поиска. `qwen2.5:3b` получает вопрос вместе с четырьмя найденными фрагментами и формирует ответ. Вся база знаний не помещается в prompt целиком.

## Установка

Создание виртуального окружения выполняете вы. Проект не создаёт и не изменяет `.venv` автоматически.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull bge-m3
```

Локальный `.env` уже создан. Он хранит URL Ollama и имена моделей, исключён из Git и изменяется при необходимости.

## Запуск

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Откройте `http://localhost:8501`, нажмите «Создать индекс» и задайте вопрос: `Какие этапы подготовки данных есть в RAG?`

## Файлы

- `src/lesson_12_frameworks_and_agents.md` — копия конспекта, включённая в проект как база знаний.
- `app.py` — браузерный интерфейс и история сообщений текущей сессии.
- `rag_service.py` — загрузка, индексация, поиск и вызов Ollama.
- `settings.py` — читает `.env` и хранит настройки.
- `data/chroma` — локальный индекс, создаваемый после нажатия кнопки.

## Следующий этап

После проверки RAG добавим Langfuse: trace на вопрос, span поиска, generation модели и dataset с LLM-as-a-judge.

## Langfuse

Для следующего этапа создайте проект в Langfuse и добавьте его ключи в локальный `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

После перезапуска Streamlit каждый ответ создаёт trace `answer_question` со span `retrieve_context` и generation `generate_answer`. В generation записываются input/output, latency и число токенов из ответа Ollama. В боковой панели доступны создание датасета `lesson_12_rag_evaluation` и запуск проверки двух вопросов. Локальная `qwen2.5:3b` выступает LLM-судьёй и записывает оценку `answer_groundedness` (0 или 1) в Langfuse.
