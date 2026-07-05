# vLLM для Tutor Bot

Tutor Bot подключается к vLLM по OpenAI-compatible API. Сервер vLLM запускается отдельно на Ubuntu, а приложение может продолжать работать на Windows.

## Требования

- Ubuntu или другой поддерживаемый Linux.
- Совместимый GPU и установленный драйвер.
- Python версии, поддерживаемой выбранным релизом vLLM.
- Достаточный объём VRAM для модели, контекста и выбранного формата весов.

Перед установкой проверьте актуальные требования в официальной документации vLLM: <https://docs.vllm.ai/en/stable/getting_started/installation/gpu/>.

## Установка и запуск

Создайте отдельное Python-окружение и установите vLLM согласно инструкции для используемого GPU. Затем задайте идентификатор модели и секретный ключ:

```bash
export VLLM_MODEL="Qwen/Qwen3-8B"
export VLLM_API_KEY="replace-with-a-secret-key"

vllm serve "$VLLM_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --generation-config vllm
```

Идентификатор модели должен совпадать со значением `VLLM_MODEL` в Tutor Bot. Для Chat Completions модель должна содержать chat template. Если его нет, передайте серверу совместимый шаблон через `--chat-template`.

Не публикуйте порт 8000 напрямую в интернете. В локальной сети ограничьте доступ firewall до IP машины с Tutor Bot. Для внешнего подключения используйте HTTPS reverse proxy или VPN.

## Настройка Tutor Bot

Добавьте в `.env`:

```dotenv
VLLM_BASE_URL=http://ubuntu-server:8000/v1
VLLM_API_KEY=replace-with-a-secret-key
VLLM_MODEL=Qwen/Qwen3-8B
```

Перезапустите Streamlit, откройте страницу `LLMs` и выберите `vLLM — Qwen/Qwen3-8B`. При необходимости сделайте этот вариант используемым по умолчанию.

## Проверка сервера

Проверьте доступность списка моделей:

```bash
curl http://ubuntu-server:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

Проверьте Chat Completions:

```bash
curl http://ubuntu-server:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Ответь одним словом: работает?"}],
    "temperature": 0.1,
    "max_tokens": 16
  }'
```

После подключения проверьте в Tutor Bot обычный Chat и генерацию задания Test Notes. В статистике токенов и Langfuse provider должен отображаться как `vllm`.

## Диагностика

- Ошибка о chat template означает, что модель не содержит подходящий шаблон. Выберите instruction/chat-модель или задайте `--chat-template`.
- CUDA out of memory требует меньшей модели, квантованных весов, меньшего контекста или изменения параметров распределения по GPU.
- `Connection refused` означает, что сервер не запущен, слушает другой адрес/порт или соединение блокирует firewall.
- HTTP 401 означает несовпадение API key.
- Ошибка загрузки модели обычно означает неверный Hugging Face model ID, отсутствие доступа к закрытой модели или недостаток дискового пространства.
- Ошибка structured output требует проверки поддержки модели и актуальной версии vLLM.
