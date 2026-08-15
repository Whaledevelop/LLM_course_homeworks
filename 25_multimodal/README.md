# MusicCaps text-to-audio search

Учебный проект мультимодального поиска аудио по текстовому описанию.

Планируемый pipeline: текстовый запрос → CLAP embedding → FAISS search → Top-K треков MusicCaps.

На текущем этапе добавлены структура проекта, общая конфигурация и зависимости. Загрузка данных, построение индекса, поиск и оценка будут реализованы на следующих этапах.

## Подготовка окружения

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CLI-заготовки запускаются из корня проекта:

```powershell
python -m scripts.download_data
python -m scripts.build_index
python -m scripts.search_demo
python -m scripts.evaluate
```

Пока они явно завершаются с `NotImplementedError`, поскольку сам pipeline ещё не реализован.
