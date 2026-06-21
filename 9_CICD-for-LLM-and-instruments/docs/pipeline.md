
## Репозиторий-проект
Проект лежит на Windows, например D:\Dev\LLM Course\9_CICD-for-LLM-and-instruments
Причины:
- Быстро открывается в Rider.
- Удобно использовать GitHub Desktop.
- Там находится Git-репозиторий.

при этом vLLM не работает на Windows, поэтому используется Ubuntu через WSL2, как сервер. Это первая консоль, запущенная в ubuntu.
vLLM лежит в Ubuntu, например ~/llm-as-judge-hw

### Установки и скачивание модели
```
python3 -m venv .venv
pip install vllm==0.7.3 openai requests pandas mlflow
pip install "transformers==4.48.3"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \  --local-dir ./models/Qwen2.5-1.5B-Instruct
```

### Запуск vLLM - сервер №1

```bash
cd ~/llm-as-judge-hw
source .venv/bin/activate
./run_vllm.sh
```

Контент run_vllm.sh, т.е. это обертка над python -m vllm.entrypoints.openai.api_server, запускающей vLLM. Содержится и в Ubunty и в репозитории (для проверки)
```
python -m vllm.entrypoints.openai.api_server \  
--model ./models/Qwen2.5-1.5B-Instruct \  
--served-model-name Qwen/Qwen2.5-1.5B-Instruct \  
--dtype float16 \  
--trust-remote-code \  
--host 0.0.0.0 \  
--port 8000 \  
--gpu-memory-utilization 0.85 \  
--max-model-len 2048 \  
--swap-space 1
```

После запуска сервер слушает по адресу `0.0.0.0:8000`, но клиенты ходят на: http://localhost:8000/v1/chat/completions

## Запуск MLFlow UI - сервер №2
```
cd ~/llm-as-judge-hw
source .venv/bin/activate
mlflow ui --host 0.0.0.0 --port 5000
```
После запуска сервер слушает по адресу `0.0.0.0:5000`, но клиенты ходят на: http://localhost:5000

## Запуск клиентских консолей

### Запуск клиенской консоли на Ubuntu.

Входим в Ubuntu через приложение

```bash
cd ~/llm-as-judge-hw
source .venv/bin/activate
cd "/mnt/d/Dev/LLM Course/9_CICD-for-LLM-and-instruments"
```

Выполнение скриптов из репозитория
```bash
python scripts/test_requests.py
python scripts/test_openai.py
python scripts/mlflow_eval.py
python scripts/mlflow_genai_eval.py
python scripts/mlflow_genai_eval_with_errors.py
```

### Запуск клиентской консоли в Windows
Возможен, но не рекомендуется, т.к. в Ubuntu уже настроен Python venv, уже установлены requests, openai, mlflow. Не нужно отдельно ставить зависимости на Windows.
Меньше проблем с версиями библиотек 

Но это возможно, т.к. клиент обращается к серверу по HTTP

## Схема

```text  
python3 -m venv .venv  
↓  
создали окружение  
  
pip install vllm  
↓  
установили vLLM  
  
python -m vllm.entrypoints.openai.api_server  
↓  
запустили сервер vLLM  
  
test_requests.py / test_openai.py  
↓  
обращаются к этому серверу
```
