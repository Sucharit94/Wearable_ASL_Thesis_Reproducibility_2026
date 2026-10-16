import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage35_evidence_freeze import build_stage35
from src.evaluation.stage35_evidence import (
    MODEL_ORDER,
    _comparison,
    _error_profile,
    _pair_row,
    percent_reduction,
    ratio,
    sha256_file,
    write_csv,
    write_json,
)


def decision_row(model_id: str, wer: float, accuracy: float, parameters: int) -> dict:
    return {
        "model_id": model_id,
        "pooled_wer": wer,
        "exact_sentence_accuracy": accuracy,
        "trainable_parameters": parameters,
        "batch1_latency_median_ms": 4.0 if model_id == "c2" else 12.0,
        "batch16_throughput_samples_per_second": 800.0 if model_id == "c2" else 100.0,
        "incremental_peak_rss_mib": 25.0 if model_id == "c2" else 70.0,
    }


class Stage35EvidenceTests(unittest.TestCase):
    def test_model_order_is_frozen(self) -> None:
        self.assertEqual(MODEL_ORDER, ("g1", "c1", "c2", "c3", "c4"))

    def test_parameter_reduction(self) -> None:
        observed = percent_reduction(4_791_688, 656_168)
        self.assertAlmostEqual(observed, 86.30612009796965)

    def test_parameter_ratio(self) -> None:
        self.assertAlmostEqual(ratio(4_791_688, 656_168), 7.302532278318967)

    def test_invalid_ratio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ratio(1.0, 0.0)

    def test_error_profile_calculates_shares(self) -> None:
        result = _error_profile(
            "c2", {"substitutions": "10", "deletions": "30", "insertions": "5"}
        )
        self.assertEqual(result["total_edits"], 45)
        self.assertEqual(result["dominant_error_type"], "deletion")
        self.assertAlmostEqual(result["deletion_share"], 30 / 45)

    def test_missing_error_counts_are_not_fabricated(self) -> None:
        result = _error_profile("c2", {})
        self.assertFalse(result["aggregate_counts_available"])
        self.assertEqual(result["total_edits"], "")
        self.assertEqual(result["dominant_error_type"], "not_available")

    def test_pair_lookup_handles_reverse_order(self) -> None:
        rows = [{"model_a": "c4", "model_b": "c2", "wer_difference_direction": "not_resolved"}]
        row, reversed_pair = _pair_row(rows, "c2", "c4")
        self.assertIsNotNone(row)
        self.assertTrue(reversed_pair)

    def test_comparison_contains_quality_and_efficiency(self) -> None:
        g1 = decision_row("g1", 0.001, 0.996, 4_791_688)
        c2 = decision_row("c2", 0.040, 0.865, 656_168)
        pairwise = [
            {
                "model_a": "g1",
                "model_b": "c2",
                "wer_difference_direction": "a_lower_wer",
                "holm_adjusted_p_value": "0.002",
            }
        ]
        result = _comparison("reference", g1, c2, pairwise)
        self.assertGreater(result["wer_delta_b_minus_a"], 0.0)
        self.assertGreater(result["parameter_reduction_a_to_b_percent"], 80.0)
        self.assertEqual(result["paired_wer_conclusion"], "a_lower_wer")

    def test_end_to_end_build_with_frozen_stage34_fixture(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in (
                "src/evaluation/stage35_evidence.py",
                "scripts/build_stage35_evidence_freeze.py",
                "scripts/verify_stage35_evidence_freeze.py",
                "tests/test_stage35_evidence_freeze.py",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative, destination)

            stage34 = root / "artifacts" / "analysis" / "stage34"
            stage34.mkdir(parents=True)
            performance_rows = []
            interval_rows = []
            efficiency_rows = []
            decision_rows = []
            values = {
                "g1": (0.001, 0.996, 0.004, 4_791_688, 12.0, 120.0, 70.0, "accuracy_reference"),
                "c1": (0.067, 0.800, 0.200, 392_488, 2.2, 1700.0, 21.0, "eligible_comparator"),
                "c2": (0.040, 0.865, 0.135, 656_168, 4.5, 856.0, 26.0, "recommended_lightweight"),
                "c3": (0.445, 0.136, 0.864, 237_352, 2.1, 1726.0, 17.0, "performance_threshold_not_met"),
                "c4": (0.041, 0.877, 0.123, 348_604, 24.9, 42.0, 100.0, "eligible_comparator"),
            }
            for model_id, value in values.items():
                wer, accuracy, ser, parameters, latency, throughput, memory, role = value
                performance_rows.append(
                    {
                        "model_id": model_id,
                        "pooled_wer": wer,
                        "exact_sentence_accuracy": accuracy,
                        "ser": ser,
                        "substitutions": 10,
                        "deletions": 30,
                        "insertions": 5,
                    }
                )
                interval_rows.append(
                    {
                        "model_id": model_id,
                        "pooled_wer_ci95_lower": max(0.0, wer - 0.001),
                        "pooled_wer_ci95_upper": wer + 0.001,
                        "exact_accuracy_ci95_lower": max(0.0, accuracy - 0.002),
                        "exact_accuracy_ci95_upper": min(1.0, accuracy + 0.002),
                    }
                )
                efficiency_rows.append(
                    {
                        "model_id": model_id,
                        "trainable_parameters": parameters,
                        "parameter_reduction_vs_g1_percent": percent_reduction(4_791_688, parameters),
                        "g1_to_model_parameter_ratio": ratio(4_791_688, parameters),
                        "serialized_state_dict_mib": parameters * 4 / 1024**2,
                        "batch1_latency_median_ms": latency,
                        "batch16_throughput_samples_per_second": throughput,
                        "incremental_peak_rss_mib": memory,
                    }
                )
                decision_rows.append(
                    {
                        "model_id": model_id,
                        "quality_eligible": model_id != "c3",
                        "lightweight_eligible": parameters <= 1_000_000,
                        "pareto_frontier": model_id != "c3",
                        "mean_scenario_score": 0.5,
                        "final_role": role,
                    }
                )

            files_by_substage = {
                "b": [("stage34b_model_metrics.csv", performance_rows)],
                "c": [
                    ("stage34c_model_confidence_intervals.csv", interval_rows),
                    (
                        "stage34c_pairwise_tests.csv",
                        [
                            {
                                "model_a": "g1",
                                "model_b": "c2",
                                "wer_difference_direction": "a_lower_wer",
                                "holm_adjusted_p_value": 0.002,
                            },
                            {
                                "model_a": "c2",
                                "model_b": "c4",
                                "wer_difference_direction": "not_resolved",
                                "holm_adjusted_p_value": 0.81,
                            },
                        ],
                    ),
                ],
                "d": [("stage34d_efficiency_metrics.csv", efficiency_rows)],
                "e": [("stage34e_decision_matrix.csv", decision_rows)],
            }
            for substage, file_rows in files_by_substage.items():
                generated = {}
                for filename, rows in file_rows:
                    path = stage34 / filename
                    write_csv(path, rows)
                    generated[path.relative_to(root).as_posix()] = sha256_file(path)
                contract_path = stage34 / f"stage34{substage}_contract.json"
                write_json(
                    contract_path,
                    {"stage": 34, "substage": substage.upper(), "generated_sha256": generated},
                )
                (stage34 / f"stage34{substage}_contract.sha256").write_text(
                    sha256_file(contract_path) + "\n", encoding="utf-8", newline="\n"
                )

            result = build_stage35(root)
            self.assertEqual(result["accuracy_reference"], "g1")
            self.assertEqual(result["recommended_lightweight"], "c2")
            self.assertTrue(result["contract_path"].is_file())
            self.assertEqual(result["contract"]["validation"]["models"], 5)


if __name__ == "__main__":
    unittest.main()
