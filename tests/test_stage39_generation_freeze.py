import unittest

from src.evaluation.stage39_generation import (
    build_prompt,
    generator_execution_template,
    nearest_candidates,
    request_id,
    validate_reference_firewall,
)


class Stage39Tests(unittest.TestCase):
    def setUp(self):
        self.bank = [
            {"sentence_id": 1, "gloss_tokens": ("your", "name", "what"), "gloss": "your name what", "english": "what is your name"},
            {"sentence_id": 2, "gloss_tokens": ("you", "from", "where"), "gloss": "you from where", "english": "where are you from"},
            {"sentence_id": 3, "gloss_tokens": ("hello",), "gloss": "hello", "english": "hello"},
        ]

    def test_nearest_candidate_exact(self):
        selected = nearest_candidates(("you", "from", "where"), self.bank, top_k=2)
        self.assertEqual(selected[0]["sentence_id"], 2)
        self.assertEqual(selected[0]["normalized_gloss_distance"], 0.0)

    def test_tie_breaks_by_sentence_id(self):
        selected = nearest_candidates(("unknown",), self.bank, top_k=3)
        self.assertEqual([row["sentence_id"] for row in selected[:2]], [3, 1])

    def test_request_id_is_deterministic(self):
        payload = {"outer_fold": 1, "model_id": "c2", "source_gloss": "hello"}
        self.assertEqual(request_id(payload), request_id(dict(reversed(list(payload.items())))))

    def test_prompt_contains_constraints(self):
        candidates = nearest_candidates(("hello",), self.bank, top_k=2)
        prompt = build_prompt("hello", candidates)
        self.assertIn("Do not invent", prompt)
        self.assertIn("Return only JSON", prompt)
        self.assertNotIn("reference_english", prompt)

    def test_firewall_accepts_training_candidate_id(self):
        row = {
            "request_id": "l2_x", "model_id": "c2", "outer_fold": 1,
            "source_gloss": "hello", "prompt": "hello",
            "expected_output_schema": {},
            "training_candidates": [{
                "rank": 1, "sentence_id": 3, "gloss": "hello",
                "english": "hello", "normalized_gloss_distance": 0.0,
            }],
        }
        validate_reference_firewall([row])

    def test_firewall_rejects_reference(self):
        with self.assertRaises(RuntimeError):
            validate_reference_firewall([{"reference_english": "secret"}])

    def test_firewall_rejects_sample_id(self):
        with self.assertRaises(RuntimeError):
            validate_reference_firewall([{"sample_id": "outer_1"}])

    def test_execution_template_is_blocked(self):
        template = generator_execution_template()
        self.assertEqual(template["status"], "BLOCKED_UNTIL_BACKEND_FROZEN")
        self.assertIsNone(template["backend"])
        self.assertFalse(template["execution"]["reference_english_available_to_generator"])
        self.assertFalse(template["execution"]["api_key_saved_in_repository"])


if __name__ == "__main__":
    unittest.main()
