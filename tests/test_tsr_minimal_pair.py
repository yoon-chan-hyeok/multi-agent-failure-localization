from __future__ import annotations

import unittest

from failure_attribution.prompts import (
    tsr_minimal_r0_prompt,
    tsr_minimal_r1_prompt,
)
from failure_attribution.schema import Case, LogStep


class TsrMinimalPairPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = Case(
            case_id="minimal-pair-test",
            problem="Find the correct item.",
            steps=[
                LogStep(step=0, agent="Planner", content="I will gather evidence."),
                LogStep(step=1, agent="WebSurfer", content="I found the wrong item."),
            ],
        )
        self.requirements = [
            {
                "id": "evidence-check",
                "description": "Verify the item using a reliable source.",
            }
        ]

    def test_both_conditions_use_agent_step_only_output(self) -> None:
        prompts = [
            tsr_minimal_r0_prompt(self.case, self.case.steps),
            tsr_minimal_r1_prompt(
                self.case,
                self.requirements,
                self.case.steps,
            ),
        ]
        for prompt in prompts:
            self.assertIn('"agent": "<agent name>"', prompt)
            self.assertIn('"step": <integer>', prompt)
            self.assertNotIn('"reason"', prompt)
            self.assertNotIn('"confidence"', prompt)
            self.assertNotIn('"error_type"', prompt)
            self.assertNotIn('"violated_constraint"', prompt)

    def test_r0_has_no_requirement_material(self) -> None:
        prompt = tsr_minimal_r0_prompt(self.case, self.case.steps).lower()
        self.assertNotIn("requirement", prompt)
        self.assertNotIn("constraint", prompt)
        self.assertNotIn("diagnostic reference", prompt)

    def test_r1_differs_only_by_requirement_treatment(self) -> None:
        r0 = tsr_minimal_r0_prompt(self.case, self.case.steps)
        r1 = tsr_minimal_r1_prompt(
            self.case,
            self.requirements,
            self.case.steps,
        )
        treatment = (
            "\nTASK-SUCCESS REQUIREMENTS:\n"
            f"{self.requirements}\n\n"
            "Use these requirements as diagnostic references.\n"
        )
        self.assertIn(treatment, r1)
        self.assertEqual(r0, r1.replace(treatment, ""))

    def test_side_information_is_absent(self) -> None:
        prompts = [
            tsr_minimal_r0_prompt(self.case, self.case.steps),
            tsr_minimal_r1_prompt(
                self.case,
                self.requirements,
                self.case.steps,
            ),
        ]
        for prompt in prompts:
            self.assertNotIn("Ground Truth Answer:", prompt)
            self.assertNotIn("Final System Answer:", prompt)


if __name__ == "__main__":
    unittest.main()
