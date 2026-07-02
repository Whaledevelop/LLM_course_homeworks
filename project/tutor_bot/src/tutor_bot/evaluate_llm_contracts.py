import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tutor_bot.application.assignment_review import AssignmentReview
from tutor_bot.application.note_metadata_suggestion import NoteMetadataSuggestion
from tutor_bot.application.recall_answer_review import RecallAnswerReview
from tutor_bot.config import get_settings


_SCHEMA_BY_SCENARIO = {
    "metadata_suggestion": NoteMetadataSuggestion,
    "assignment_review": AssignmentReview,
    "active_recall_review": RecallAnswerReview,
}


class LlmContractCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    should_be_valid: bool
    output: dict[str, Any]


class LlmContractDataset(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    cases: tuple[LlmContractCase, ...] = Field(min_length=1)


def main() -> int:
    arguments = _parse_arguments()
    dataset_path = (
        arguments.dataset_path or get_settings().evaluation_dir / "llm_contract_cases.json"
    )
    dataset = LlmContractDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    report = _evaluate(dataset)

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    if report["failed_case_count"] > 0:
        return 1

    return 0


def _evaluate(
    dataset: LlmContractDataset,
) -> dict[str, Any]:
    failures = []
    case_counts_by_scenario: dict[str, int] = {}

    for contract_case in dataset.cases:
        case_counts_by_scenario[contract_case.scenario] = (
            case_counts_by_scenario.get(contract_case.scenario, 0) + 1
        )
        is_valid, error_type = _validate_case(contract_case)

        if is_valid != contract_case.should_be_valid:
            failures.append(
                {
                    "id": contract_case.id,
                    "scenario": contract_case.scenario,
                    "should_be_valid": contract_case.should_be_valid,
                    "was_valid": is_valid,
                    "error_type": error_type,
                }
            )

    return {
        "case_count": len(dataset.cases),
        "passed_case_count": len(dataset.cases) - len(failures),
        "failed_case_count": len(failures),
        "case_counts_by_scenario": case_counts_by_scenario,
        "failures": failures,
    }


def _validate_case(
    contract_case: LlmContractCase,
) -> tuple[bool, str | None]:
    schema = _SCHEMA_BY_SCENARIO.get(contract_case.scenario)

    if schema is None:
        return False, "UnknownScenario"

    try:
        schema.model_validate(contract_case.output)
    except ValidationError:
        return False, "ValidationError"

    return True, None


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
