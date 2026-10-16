import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.stage37_language_protocol import (
    arm_rows,
    build_fold_plan,
    deterministic_retrieval,
    leakage_rows,
    metric_rows,
    portable_text_sha256,
    prompt_contract,
    require_portable_recorded_hash,
    system_rows,
    validate_language_config,
    validate_targets,
    word_edit_distance,
)


class Stage37ProtocolTests(unittest.TestCase):
    def test_portable_hash_ignores_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.txt"
            right = Path(directory) / "right.txt"
            left.write_bytes(b"a\nb\n")
            right.write_bytes(b"a\r\nb\r\n")
            self.assertEqual(portable_text_sha256(left), portable_text_sha256(right))

    def test_recorded_hash_accepts_crlf_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "x.csv"
            digest = Path(directory) / "x.sha256"
            data.write_bytes(b"a,b\r\n1,2\r\n")
            digest.write_text(hashlib.sha256(b"a,b\n1,2\n").hexdigest() + "\n")
            require_portable_recorded_hash(data, digest)

    def test_language_config_firewall(self):
        config = {
            "protocol": {
                "recognition_test_predictions_are_immutable_inputs": True,
                "retrieval_and_prompt_library": "training_partition_only",
                "inputs": ["predicted_gloss", "oracle_gloss"],
                "systems": [
                    "L0_raw_gloss",
                    "L1_training_only_retrieval",
                    "L2_constrained_generator",
                    "L3_template_holdout",
                ],
            }
        }
        validate_language_config(config)

    def test_language_config_rejects_outer_library(self):
        config = {
            "protocol": {
                "recognition_test_predictions_are_immutable_inputs": True,
                "retrieval_and_prompt_library": "all_data",
                "inputs": ["predicted_gloss", "oracle_gloss"],
                "systems": [],
            }
        }
        with self.assertRaises(RuntimeError):
            validate_language_config(config)

    def test_targets_validate_token_count(self):
        rows = [{
            "sentence_id": "1", "gloss_raw": "your name what",
            "english_raw": "what is your name", "gloss_token_count": "3",
            "gloss_tokens_json": json.dumps(["your", "name", "what"]),
        }]
        validate_targets(rows, expected_count=1)

    def test_word_edit_distance(self):
        self.assertEqual(word_edit_distance(["your", "name", "what"], ["what", "name"]), 2)
        self.assertEqual(word_edit_distance(["hello"], ["hello"]), 0)

    def test_retrieval_is_deterministic(self):
        candidates = [
            {"sentence_id": 8, "gloss_tokens": ["you", "fine"]},
            {"sentence_id": 2, "gloss_tokens": ["you", "well"]},
        ]
        selected = deterministic_retrieval(["you", "okay"], candidates)
        self.assertEqual(selected["sentence_id"], 2)

    def test_fold_plan_enforces_one_outer_test(self):
        rows = []
        for sample_index in (1, 2):
            for fold in (1, 2):
                rows.append({
                    "sample_id": f"s{sample_index}",
                    "sentence_id": str(sample_index),
                    "stage32_outer_fold": str(fold),
                    "partition": "outer_test" if fold == sample_index else "inner_train",
                })
        plan = build_fold_plan(rows, expected_folds=2)
        self.assertEqual(len(plan), 2)
        self.assertTrue(all(row["outer_samples_in_retrieval_bank"] == 0 for row in plan))

    def test_fold_plan_rejects_repeated_outer_test(self):
        rows = [
            {"sample_id": "s1", "sentence_id": "1", "stage32_outer_fold": "1", "partition": "outer_test"},
            {"sample_id": "s1", "sentence_id": "1", "stage32_outer_fold": "2", "partition": "outer_test"},
        ]
        with self.assertRaises(RuntimeError):
            build_fold_plan(rows, expected_folds=2)

    def test_frozen_tables_have_expected_scope(self):
        self.assertEqual(len(system_rows()), 4)
        self.assertEqual(len(arm_rows()), 6)
        self.assertEqual(len(metric_rows()), 8)
        self.assertEqual(len(leakage_rows()), 7)
        self.assertFalse(prompt_contract()["outer_reference_in_prompt"])
        self.assertEqual(prompt_contract()["decoding"]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
