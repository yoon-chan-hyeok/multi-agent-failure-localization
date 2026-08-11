from __future__ import annotations

import unittest

from failure_attribution.prompts import (
    ccv_requirements_direct_prompt,
    tsr_direct_no_requirements_prompt,
)
from failure_attribution.schema import Case, LogStep


class TsrDirectRequirementAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = Case(
            case_id="direct-ablation-test",
            problem="Find the correct item.",
            steps=[
                LogStep(step=0, agent="Planner", content="I will gather evidence."),
                LogStep(step=1, agent="WebSurfer", content="I found the wrong item."),
            ],
        )
        self.requirements = [
            {
                "id": "evidence-check",
                "type": "evidence",
                "description": "Verify the item using a reliable source.",
                "violation_criteria": "The selected item is unsupported.",
            }
        ]

    def test_r0_contains_no_requirement_specific_language(self) -> None:
        prompt = tsr_direct_no_requirements_prompt(self.case, self.case.steps)
        lowered = prompt.lower()
        self.assertNotIn("requirement", lowered)
        self.assertNotIn("constraint", lowered)
        self.assertNotIn("violation", lowered)

    def test_r0_preserves_generic_direct_rules(self) -> None:
        prompt = tsr_direct_no_requirements_prompt(self.case, self.case.steps)
        self.assertIn("Base the final attribution on evidence in the full conversation.", prompt)
        self.assertIn("Prefer the earliest responsible error over a later downstream consequence.", prompt)
        self.assertIn("Do not attribute the failure to the human/user/problem statement.", prompt)
        self.assertIn("Choose an action taken by the multi-agent system.", prompt)

    def test_r1_retains_only_the_expected_requirement_specific_material(self) -> None:
        prompt = ccv_requirements_direct_prompt(
            self.case,
            self.requirements,
            self.case.steps,
        )
        self.assertIn("requirement-guided direct failure attribution", prompt)
        self.assertIn("Constraints:", prompt)
        self.assertIn("evidence-check", prompt)
        self.assertIn('"violated_constraint"', prompt)

    def test_both_conditions_hide_side_information_when_case_is_sanitized(self) -> None:
        r1 = ccv_requirements_direct_prompt(
            self.case,
            self.requirements,
            self.case.steps,
        )
        r0 = tsr_direct_no_requirements_prompt(self.case, self.case.steps)
        for prompt in (r1, r0):
            self.assertNotIn("Ground Truth Answer:", prompt)
            self.assertNotIn("Final System Answer:", prompt)
            self.assertIn('"step": <integer step number from the conversation>', prompt)
            self.assertIn('"agent": "<agent name>"', prompt)


if __name__ == "__main__":
    unittest.main()
