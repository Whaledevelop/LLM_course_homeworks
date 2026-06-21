import argparse

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question for local vLLM model")
    args = parser.parse_args()

    client = OpenAI(
        api_key="DUMMY",
        base_url="http://localhost:8000/v1",
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {
                "role": "user",
                "content": args.question
            }
        ],
        temperature=0.1,
        max_tokens=100,
    )

    answer = response.choices[0].message.content

    print("Question:", args.question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()