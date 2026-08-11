from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path

from failure_attribution.prompts import a2p_repo_exact_prompt
from failure_attribution.schema import Case, LogStep


ROOT = Path(__file__).resolve().parents[1]
A2P_REPO = ROOT / ".external" / "A2P"
A2P_COMMIT = "7953d780c85054721a7b4bf246bcf60a16bb28af"


class A2PRepoExactPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = Case(
            case_id="prompt-test",
            problem="Find the answer.",
            ground_truth="42",
            steps=[
                LogStep(step=7, agent="AgentA", content="First action."),
                LogStep(step=9, agent="AgentB", content="Second action."),
            ],
        )

    def test_prompt_matches_repository_function_at_pinned_commit(self) -> None:
        if not (A2P_REPO / ".git").exists():
            self.skipTest("Pinned A2P repository checkout is unavailable.")

        safe_directory = str(A2P_REPO.resolve()).replace("\\", "/")
        source = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={safe_directory}",
                "-C",
                str(A2P_REPO),
                "show",
                f"{A2P_COMMIT}:Automated_FA/Lib/utils.py",
            ],
            text=True,
            encoding="utf-8",
        )
        module = ast.parse(source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "construct_a2p_prompt"
        )
        namespace: dict[str, object] = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<a2p-repo>", "exec"), namespace)
        repository_prompt = namespace["construct_a2p_prompt"](
            problem=self.case.problem,
            ground_truth=self.case.ground_truth,
            chat_content="unused",
            chat_history=[
                {"name": step.agent, "content": step.content}
                for step in self.case.steps
            ],
            index_agent="name",
            a2p=True,
        )

        self.assertEqual(a2p_repo_exact_prompt(self.case), repository_prompt)

    def test_uses_repository_zero_based_contextual_indices(self) -> None:
        prompt = a2p_repo_exact_prompt(self.case)
        self.assertIn("Step 0 - AgentA: First action.", prompt)
        self.assertIn("Step 1 - AgentB: Second action.", prompt)
        self.assertNotIn("downstream symptom", prompt)


if __name__ == "__main__":
    unittest.main()
