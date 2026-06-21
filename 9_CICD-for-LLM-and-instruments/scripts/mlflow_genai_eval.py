import os

import mlflow
import pandas as pd
from mlflow.metrics.genai import EvaluationExample, make_genai_metric
from openai import OpenAI


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_URL = "http://localhost:8000/v1"

os.environ["OPENAI_API_KEY"] = "dummy"

client = OpenAI(
    api_key="dummy",
    base_url=BASE_URL,
)


def ask_local_model(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}],
        temperature=0.1,
        max_tokens=100,
    )
    return response.choices[0].message.content


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("local-vllm-genai-judge")

    eval_data = pd.DataFrame(
        [
            {
                "inputs": "What is the capital of Germany?",
                "targets": "Berlin",
            },
            {
                "inputs": "What is the capital of England?",
                "targets": "London",
            },
            {
                "inputs": "What is the capital of France?",
                "targets": "Paris",
            },
        ]
    )

    eval_data["predictions"] = eval_data["inputs"].apply(ask_local_model)

    example = EvaluationExample(
        input="What is the capital of Germany?",
        output="The capital of Germany is Berlin.",
        score=5,
        justification="The answer correctly identifies Berlin as the capital of Germany.",
        grading_context={"targets": "Berlin"},
    )

    answer_correctness = make_genai_metric(
        name="local_answer_correctness",
        definition=(
            "Evaluates whether the model output is factually correct "
            "compared with the reference answer."
        ),
        grading_prompt=(
            "Score the output from 1 to 5.\n"
            "Score 1: The output is completely incorrect.\n"
            "Score 2: The output is mostly incorrect.\n"
            "Score 3: The output is partially correct.\n"
            "Score 4: The output is mostly correct with minor issues.\n"
            "Score 5: The output is fully correct and matches the reference answer."
        ),
        examples=[example],
        model=f"openai:/{MODEL_NAME}",
        grading_context_columns=["targets"],
        parameters={"temperature": 0.0, "max_tokens": 200},
        aggregations=["mean", "variance"],
        greater_is_better=True,
        proxy_url="http://localhost:8000/v1/chat/completions",
        extra_headers={"Authorization": "Bearer dummy"},
        max_workers=1,
    )

    with mlflow.start_run(run_name="qwen-local-genai-judge"):
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("base_url", BASE_URL)
        mlflow.log_param("judge_metric", "make_genai_metric")

        result = mlflow.evaluate(
            data=eval_data,
            predictions="predictions",
            targets="targets",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[answer_correctness],
        )

        print(eval_data)
        print(result.metrics)


if __name__ == "__main__":
    main()