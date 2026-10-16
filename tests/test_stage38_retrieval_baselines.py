import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.evaluation.stage38_retrieval import (
    PredictionSource,
    content_tokens,
    edit_counts,
    locate_prediction_source,
    multiset_f1,
    parse_jsonl,
    summarize_outputs,
    tokenize_text,
)


class Stage38Tests(unittest.TestCase):
    def test_tokenization_expands_gloss_underscore(self):
        self.assertEqual(tokenize_text("WHATS_UP friend's"), ("whats", "up", "friend's"))

    def test_edit_counts_exact(self):
        self.assertEqual(edit_counts(("what", "is", "your", "name"), ("what", "is", "your", "name")), (0, 0, 0))

    def test_edit_counts_deletion(self):
        self.assertEqual(edit_counts(("i", "need", "water"), ("i", "water")), (0, 1, 0))

    def test_edit_counts_insertion(self):
        self.assertEqual(edit_counts(("i", "go"), ("i", "will", "go")), (0, 0, 1))

    def test_content_tokens_remove_grammar_words(self):
        self.assertEqual(content_tokens(("where", "is", "hospital")), ("hospital",))

    def test_multiset_f1(self):
        self.assertAlmostEqual(multiset_f1(("milk", "cookie"), ("milk",)), 2 / 3)

    def test_parse_jsonl(self):
        rows = parse_jsonl(b'{"sample_id":"a"}\n{"sample_id":"b"}\n')
        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])

    def test_locate_prediction_in_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            imports = root / "imports"
            imports.mkdir()
            archive = imports / "stage32c_results_without_checkpoints.zip"
            payload = b'{"sample_id":"x"}\n'
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("artifacts/runs/known_signer/fold01/c2/predictions.jsonl", payload)
            source = locate_prediction_source(root, "c2", 1)
            self.assertEqual(source.data, payload)
            self.assertIn("::", source.location)

    def test_conflicting_prediction_copies_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "artifacts/runs/known_signer/fold01/g1"
            local.mkdir(parents=True)
            (local / "predictions.jsonl").write_bytes(b"one\n")
            imports = root / "imports"
            imports.mkdir()
            with zipfile.ZipFile(imports / "copy.zip", "w") as handle:
                handle.writestr("artifacts/runs/known_signer/fold01/g1/predictions.jsonl", b"two\n")
            with self.assertRaises(RuntimeError):
                locate_prediction_source(root, "g1", 1)

    def test_summary_has_two_systems(self):
        base = {
            "sample_id": "s1", "model_id": "g1", "reference_words": 1,
            "content_token_f1": 1.0, "unsupported_content_tokens": 0,
            "output_content_tokens": 1, "substitutions": 0, "deletions": 0,
            "insertions": 0, "edit_distance": 0, "exact_match": True,
        }
        rows = []
        for index, model_id in enumerate(("g1", "c1", "c2", "c3", "c4"), 1):
            rows.append({**base, "sample_id": f"s{index}", "model_id": model_id, "system_id": "L0_raw_gloss"})
            rows.append({**base, "sample_id": f"s{index}", "model_id": model_id, "system_id": "L1_training_only_retrieval"})
        self.assertEqual(len(summarize_outputs(rows)), 10)


if __name__ == "__main__":
    unittest.main()
