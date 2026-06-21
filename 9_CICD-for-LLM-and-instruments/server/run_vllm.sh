python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dtype float16 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000