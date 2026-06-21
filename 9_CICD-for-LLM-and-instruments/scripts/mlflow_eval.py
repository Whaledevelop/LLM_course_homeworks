import pandas as pd
import mlflow
from openai import OpenAI


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_URL = "http://localhost:8000/v1"


client = OpenAI(
    api_key="dummy",
    base_url=BASE_URL,
)


def ask_local_model(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        temperature=0.1,
        max_tokens=100,
    )

    return response.choices[0].message.content


def simple_judge_score(prediction: str, reference: str) -> int:
    prediction_lower = prediction.lower()
    reference_lower = reference.lower()

    if reference_lower in prediction_lower:
        return 1

    return 0


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("local-vllm-judge")

    data = pd.DataFrame(
        [
            {
                "question": "What is the capital of Germany?",
                "reference": "Berlin",
            },
            {
                "question": "What is the capital of England?",
                "reference": "London",
            },
            {
                "question": "What is the capital of France?",
                "reference": "Paris",
            },
        ]
    )

    results = []

    with mlflow.start_run(run_name="qwen-local-basic-eval"):
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("base_url", BASE_URL)
        mlflow.log_param("judge_type", "simple_reference_match")

        for _, row in data.iterrows():
            prediction = ask_local_model(row["question"])
            score = simple_judge_score(prediction, row["reference"])

            results.append(
                {
                    "question": row["question"],
                    "reference": row["reference"],
                    "prediction": prediction,
                    "score": score,
                }
            )

        results_df = pd.DataFrame(results)

        average_score = results_df["score"].mean()

        mlflow.log_metric("average_score", average_score)
        mlflow.log_table(results_df, "evaluation_results.json")

        print(results_df)
        print("Average score:", average_score)


if __name__ == "__main__":
    main()