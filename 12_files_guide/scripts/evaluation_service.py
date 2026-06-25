from langfuse import Evaluation
from langfuse.api import NotFoundError

from llm_service import generate_chat_response
from observability import get_langfuse_client
from rag_service import answer_question
from settings import Settings


DATASET_NAME = "knowledge_base_evaluation"

EVALUATION_CASES = (
    {
        "question": "Какая основная тема загруженных документов?",
        "expected_answer": "Ответ должен кратко описывать тему, опираясь только на загруженные документы.",
    },
    {
        "question": "Какие ключевые понятия упоминаются в загруженных документах?",
        "expected_answer": "Ответ должен перечислять понятия, которые действительно есть в загруженных документах.",
    },
)


def create_evaluation_dataset(settings: Settings) -> bool:
    langfuse_client = get_langfuse_client(settings)
    if langfuse_client is None:
        raise RuntimeError("Добавьте ключи Langfuse в .env.")

    try:
        langfuse_client.get_dataset(DATASET_NAME)

        return False
    except NotFoundError:
        langfuse_client.create_dataset(
            name=DATASET_NAME,
            description="Проверка ответов справочника по загруженным документам.",
        )

    for evaluation_case in EVALUATION_CASES:
        langfuse_client.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"question": evaluation_case["question"]},
            expected_output=evaluation_case["expected_answer"],
        )

    langfuse_client.flush()

    return True


def run_evaluation(settings: Settings) -> list[dict]:
    langfuse_client = get_langfuse_client(settings)
    if langfuse_client is None:
        raise RuntimeError("Добавьте ключи Langfuse в .env.")

    dataset = langfuse_client.get_dataset(DATASET_NAME)
    experiment = dataset.run_experiment(
        name="knowledge_base_rag_evaluation",
        description="Проверка ответов RAG по контрольным вопросам датасета.",
        task=lambda *, item: _answer_dataset_item(item, settings),
        evaluators=[
            lambda *, input, output, expected_output, metadata: _evaluate_answer(
                input,
                output,
                expected_output,
                settings,
            )
        ],
        max_concurrency=1,
        metadata={"model": settings.chat_model, "model_cost": "0"},
    )
    results = [
        {
            "question": item_result.item.input["question"],
            "score": item_result.evaluations[0].value,
            "answer": item_result.output,
        }
        for item_result in experiment.item_results
    ]

    langfuse_client.flush()

    return results


def _answer_dataset_item(item, settings: Settings) -> str:
    answer = answer_question(item.input["question"], settings)

    return answer.content


def _evaluate_answer(
    input_data: dict,
    output: str,
    expected_output: str,
    settings: Settings,
) -> Evaluation:
    user_message = f"Вопрос: {input_data['question']}\nКритерий: {expected_output}\nОтвет: {output}"
    response = generate_chat_response(
        messages=[
            {
                "role": "system",
                "content": "Оцени ответ строго по критерию. Верни только 1, если критерий выполнен, иначе только 0.",
            },
            {"role": "user", "content": user_message},
        ],
        settings=settings,
        max_tokens=10,
    )
    score = 0
    if response.content.strip().startswith("1"):
        score = 1

    comment = "Ответ основан на содержимом загруженных документов."
    if not score:
        comment = "Ответ не соответствует критерию датасета."

    return Evaluation(
        name="answer_groundedness",
        value=score,
        comment=comment,
    )
