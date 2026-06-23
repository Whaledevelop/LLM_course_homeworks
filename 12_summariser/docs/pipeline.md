# Pipeline

## Требования

- Python 3.11 или новее.
- Установленный Ollama с моделью `qwen2.5:3b`.
- Установленный Tesseract OCR с языковыми пакетами `rus` и `eng`. Он нужен для определения изображений, содержащих только текст.

## Установка

В PowerShell из папки проекта (ollama включена):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
Copy-Item .env.example .env
```
## Запуск

```powershell
python -m streamlit run scripts/app.py
```

Открывает `http://localhost:8501`.

## Использование

1. Загрузите один или несколько PDF в центре страницы.
2. Оставьте переключатель `Meta Data Files` выключенным, чтобы исключить всю информацию, не относящуюся непосредственно к предмету лекции: преподавателя, программу и карту курса, цели, задания, вопросы и организационные разделы. Включите его, чтобы сохранить все страницы.
3. Выключите `Import Images`, если в Markdown нужен только текст. В этом режиме изображения не извлекаются, не обрабатываются OCR и не сохраняются.
4. Укажите папку сохранения и нажмите «Создать Markdown».
