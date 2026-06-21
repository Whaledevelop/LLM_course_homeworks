import requests

url = "http://localhost:8000/v1/chat/completions"

question = "What is the capital of Germany?";
payload = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
}

response = requests.post(url, json=payload)


answer = response.json()["choices"][0]["message"]["content"]

print("Question:", question)
print("Answer:", answer)