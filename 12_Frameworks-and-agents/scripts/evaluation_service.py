from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langfuse.api import NotFoundError

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

    results = []
    for evaluation_case in EVALUATION_CASES:
        with langfuse_client.start_as_current_observation(
            name="llm_judge_evaluation",
            as_type="evaluator",
            input={
                "question": evaluation_case["question"],
                "expected_answer": evaluation_case["expected_answer"],
            },
            metadata={"dataset_name": DATASET_NAME},
        ) as evaluation_observation:
            answer = answer_question(evaluation_case["question"], settings)
            score = _judge_answer(evaluation_case, answer.content, settings)
            langfuse_client.score_current_trace(
                name="answer_groundedness",
                value=score,
                data_type="NUMERIC",
            )
            evaluation_observation.update(
                output={"answer": answer.content, "score": score},
        )
        results.append(
            {
                "question": evaluation_case["question"],
                "score": score,
                "answer": answer.content,
            }
        )

    langfuse_client.flush()

    return results


def _judge_answer(evaluation_case: dict, answer: str, settings: Settings) -> int:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Оцени ответ строго по критерию. Верни только 1, если критерий выполнен, иначе только 0.",
            ),
            (
                "human",
                "Вопрос: {question}\nКритерий: {expected_answer}\nОтвет: {answer}",
            ),
        ]
    )
    model = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    response = (prompt | model).invoke(
        {
            "question": evaluation_case["question"],
            "expected_answer": evaluation_case["expected_answer"],
            "answer": answer,
        }
    )
    score = 0
    if str(response.content).strip().startswith("1"):
        score = 1

    return score
