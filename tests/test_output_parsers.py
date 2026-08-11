import unittest

from failure_attribution.methods import parse_a2p_official_response


class OfficialOutputParserTests(unittest.TestCase):
    def test_parses_labeled_official_output(self) -> None:
        parsed = parse_a2p_official_response(
            "Agent Name: DataAnalysis_Expert\n"
            "Step Number: 3\n"
            "Reason for Mistake: The extraction failed."
        )

        self.assertEqual(parsed["agent"], "DataAnalysis_Expert")
        self.assertEqual(parsed["step"], 3)
        self.assertNotIn("agent_recovered_from_unlabeled_lead", parsed)

    def test_recovers_unlabeled_agent_on_first_line(self) -> None:
        parsed = parse_a2p_official_response(
            "DataAnalysis_Expert:\n"
            "Step Number: 3\n"
            "Reason for Mistake: The extraction failed."
        )

        self.assertEqual(parsed["agent"], "DataAnalysis_Expert")
        self.assertEqual(parsed["step"], 3)
        self.assertTrue(parsed["agent_recovered_from_unlabeled_lead"])

    def test_parses_numbered_official_fields(self) -> None:
        parsed = parse_a2p_official_response(
            "1. Agent Name: Bioinformatics_Expert\n"
            "2. Step Number: 4\n"
            "3. Reason for Mistake: The wrong source was selected."
        )

        self.assertEqual(parsed["agent"], "Bioinformatics_Expert")
        self.assertEqual(parsed["step"], 4)
        self.assertEqual(parsed["reason"], "The wrong source was selected.")

    def test_does_not_treat_standard_field_as_agent(self) -> None:
        parsed = parse_a2p_official_response(
            "Step Number: 3\n"
            "Reason for Mistake: The extraction failed."
        )

        self.assertIsNone(parsed["agent"])
        self.assertEqual(parsed["step"], 3)


if __name__ == "__main__":
    unittest.main()
