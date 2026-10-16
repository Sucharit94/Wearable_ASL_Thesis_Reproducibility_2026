import unittest
import xml.etree.ElementTree as ET

from src.evaluation.stage36_diagnostics import (
    build_error_rows,
    build_stability_rows,
    c2_c4_fold_comparison,
    fold_wer_svg,
    normalize_fold_rows,
    quality_efficiency_svg,
)


def fixture_rows() -> list[dict]:
    wers = {
        "g1": [0.0023, 0.0004, 0.0016, 0.0007, 0.0004],
        "c1": [0.0710, 0.0551, 0.0700, 0.0673, 0.0727],
        "c2": [0.0394, 0.0406, 0.0451, 0.0388, 0.0363],
        "c3": [0.4397, 0.4554, 0.4449, 0.4386, 0.4447],
        "c4": [0.0451, 0.0471, 0.0456, 0.0353, 0.0315],
    }
    rows = []
    for fold in range(1, 6):
        for model_id, values in wers.items():
            wer = values[fold - 1]
            rows.append(
                {
                    "model_id": model_id,
                    "outer_fold": fold,
                    "pooled_wer": wer,
                    "exact_sentence_accuracy": max(0.0, 1.0 - 3 * wer),
                    "ser": min(1.0, 3 * wer),
                    "substitutions": 5,
                    "deletions": 30 if model_id == "c3" else 10,
                    "insertions": 2,
                }
            )
    return rows


class Stage36Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.fold_rows = normalize_fold_rows(fixture_rows())

    def test_exact_model_fold_matrix_is_required(self) -> None:
        with self.assertRaises(RuntimeError):
            normalize_fold_rows(fixture_rows()[:-1])

    def test_rows_are_fold_major(self) -> None:
        self.assertEqual(self.fold_rows[0]["model_id"], "g1")
        self.assertEqual(self.fold_rows[0]["outer_fold"], 1)
        self.assertEqual(self.fold_rows[-1]["model_id"], "c4")
        self.assertEqual(self.fold_rows[-1]["outer_fold"], 5)

    def test_g1_wins_all_folds(self) -> None:
        stability = {
            row["model_id"]: row for row in build_stability_rows(self.fold_rows)
        }
        self.assertEqual(stability["g1"]["fold_wins"], 5)

    def test_sample_standard_deviation_is_positive(self) -> None:
        stability = build_stability_rows(self.fold_rows)
        self.assertTrue(
            all(float(row["pooled_wer_sample_sd"]) > 0 for row in stability)
        )

    def test_error_composition_identifies_deletions(self) -> None:
        errors = {
            row["model_id"]: row for row in build_error_rows(self.fold_rows)
        }
        self.assertEqual(errors["c3"]["dominant_error_type"], "deletion")
        self.assertGreater(errors["c3"]["deletion_share"], 0.5)

    def test_error_counts_can_use_frozen_stage35_fallback(self) -> None:
        rows = fixture_rows()
        for row in rows:
            row.pop("substitutions")
            row.pop("deletions")
            row.pop("insertions")
        normalized = normalize_fold_rows(rows)
        fallback = [
            {
                "model_id": model_id,
                "substitutions": 5,
                "deletions": 25,
                "insertions": 2,
            }
            for model_id in ("g1", "c1", "c2", "c3", "c4")
        ]
        errors = {
            row["model_id"]: row
            for row in build_error_rows(normalized, fallback)
        }
        self.assertEqual(errors["c3"]["dominant_error_type"], "deletion")
        self.assertEqual(
            errors["c3"]["error_count_source"],
            "stage35_aggregate_error_profile",
        )

    def test_c2_and_c4_exchange_fold_advantages(self) -> None:
        comparison = c2_c4_fold_comparison(self.fold_rows)
        self.assertEqual(comparison["c2_lower_wer_folds"], 3)
        self.assertEqual(comparison["c4_lower_wer_folds"], 2)
        self.assertEqual(
            comparison["statistical_conclusion"],
            "not_resolved_at_95_percent",
        )

    def test_fold_svg_is_valid_xml(self) -> None:
        svg = fold_wer_svg(self.fold_rows)
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertNotIn("nan", svg.lower())

    def test_quality_efficiency_svg_is_valid_xml(self) -> None:
        rows = [
            {
                "model_id": model_id,
                "trainable_parameters": parameters,
                "pooled_wer": wer,
            }
            for model_id, parameters, wer in (
                ("g1", 4_791_688, 0.001),
                ("c1", 392_488, 0.067),
                ("c2", 656_168, 0.040),
                ("c3", 237_352, 0.445),
                ("c4", 348_604, 0.041),
            )
        ]
        svg = quality_efficiency_svg(rows)
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("C2", svg)


if __name__ == "__main__":
    unittest.main()
