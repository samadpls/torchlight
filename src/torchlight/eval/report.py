from collections import defaultdict
from statistics import mean

from torchlight.eval.apis import invalid_names, torch_names
from torchlight.eval.batches import Grade
from torchlight.eval.generate import Answer
from torchlight.schemas import EvalExample

ARM_NAMES = {
    "A": "Phi-4-mini (base model)",
    "B": "Phi-4-mini + forum search",
    "D": "torchlight (fine-tuned)",
    "E": "torchlight + forum search",
}
# Arm C, a larger model with the same search tool, is graded alongside as a reference for the
# judge but left out of the table, which compares models of the same size.
REPORTED_ARMS = ["A", "B", "D", "E"]


def _share(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.0f}%" if values else "n/a"


def _count(values: list[bool]) -> str:
    """'9 of 10' rather than a percentage, for groups too small for one to mean much."""
    return f"{sum(values)} of {len(values)}"


def results_table(examples: list[EvalExample], answers: list[Answer], grades: list[Grade]) -> str:
    needs_details = {e.thread_id: e.needs_clarification for e in examples}
    by_arm: dict[str, list[Answer]] = defaultdict(list)
    for answer in answers:
        by_arm[answer.arm].append(answer)
    grades_by_arm: dict[str, list[Grade]] = defaultdict(list)
    for grade in grades:
        grades_by_arm[grade.arm].append(grade)

    rows = [
        "| Model | Average grade (out of 5) | Graded 4 or 5 | Wrong answers (graded 1) | "
        "Code with only real torch APIs | "
        "Asked for details, when they were missing | Asked for details, when not needed | "
        "Searches per question |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in [a for a in REPORTED_ARMS if a in by_arm]:
        arm_answers = by_arm[arm]
        arm_grades = grades_by_arm[arm]
        with_code = [a for a in arm_answers if torch_names(a.answer)]
        rows.append(
            "| "
            + " | ".join(
                [
                    ARM_NAMES[arm],
                    f"{mean(g.score for g in arm_grades):.2f}" if arm_grades else "n/a",
                    _share([g.score >= 4 for g in arm_grades]),
                    _share([g.score == 1 for g in arm_grades]),
                    _count([not invalid_names(a.answer) for a in with_code]),
                    _count([g.asks for g in arm_grades if needs_details[g.thread_id]]),
                    _share([g.asks for g in arm_grades if not needs_details[g.thread_id]]),
                    f"{mean(len(a.searches) for a in arm_answers):.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)
