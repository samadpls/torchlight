from collections import defaultdict
from statistics import mean

from torchlight.eval.apis import invalid_names, torch_names
from torchlight.eval.batches import Grade
from torchlight.eval.generate import Answer
from torchlight.schemas import EvalExample

ARM_NAMES = {
    "A": "Phi-4-mini (base model)",
    "B": "Phi-4-mini + forum search",
    "C": "Claude Sonnet 5.5 + forum search (reference)",
    "D": "torchlight (fine-tuned)",
    "E": "torchlight + forum search",
}
# Same-size models first; the larger model is a ceiling, not a competitor.
ARM_ORDER = ["A", "B", "D", "E", "C"]


def _share(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.0f}%" if values else "n/a"


def results_table(examples: list[EvalExample], answers: list[Answer], grades: list[Grade]) -> str:
    needs_details = {e.thread_id: e.needs_clarification for e in examples}
    by_arm: dict[str, list[Answer]] = defaultdict(list)
    for answer in answers:
        by_arm[answer.arm].append(answer)
    grades_by_arm: dict[str, list[Grade]] = defaultdict(list)
    for grade in grades:
        grades_by_arm[grade.arm].append(grade)

    rows = [
        "| Model | Answers | Score (1-5) | Scored 4+ | Valid torch APIs | "
        "Asks when details are missing | Asks when answerable | Searches per question |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in sorted(by_arm, key=lambda a: (ARM_ORDER + sorted(by_arm)).index(a)):
        arm_answers = by_arm[arm]
        arm_grades = grades_by_arm[arm]
        with_code = [a for a in arm_answers if torch_names(a.answer)]
        rows.append(
            "| "
            + " | ".join(
                [
                    ARM_NAMES.get(arm, arm),
                    str(len(arm_answers)),
                    f"{mean(g.score for g in arm_grades):.2f}" if arm_grades else "n/a",
                    _share([g.score >= 4 for g in arm_grades]),
                    _share([not invalid_names(a.answer) for a in with_code]),
                    _share([g.asks for g in arm_grades if needs_details[g.thread_id]]),
                    _share([g.asks for g in arm_grades if not needs_details[g.thread_id]]),
                    f"{mean(len(a.searches) for a in arm_answers):.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)
