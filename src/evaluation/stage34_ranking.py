"""Transparent multi-criteria model selection for Stage 34E."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.evaluation.stage34_analysis import read_csv, write_csv


MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")
QUALITY_THRESHOLDS = {
    "maximum_pooled_wer": 0.10,
    "minimum_exact_sentence_accuracy": 0.75,
}
LIGHTWEIGHT_PARAMETER_LIMIT = 1_000_000
CRITERIA_DIRECTIONS = {
    "pooled_wer": "minimize",
    "exact_sentence_accuracy": "maximize",
    "trainable_parameters": "minimize",
    "batch1_latency_median_ms": "minimize",
    "batch16_throughput_samples_per_second": "maximize",
    "incremental_peak_rss_mib": "minimize",
}
WEIGHT_SCENARIOS = {
    "quality_priority": {
        "pooled_wer": 0.40,
        "exact_sentence_accuracy": 0.30,
        "trainable_parameters": 0.10,
        "batch1_latency_median_ms": 0.10,
        "batch16_throughput_samples_per_second": 0.05,
        "incremental_peak_rss_mib": 0.05,
    },
    "balanced": {
        "pooled_wer": 0.25,
        "exact_sentence_accuracy": 0.25,
        "trainable_parameters": 0.20,
        "batch1_latency_median_ms": 0.15,
        "batch16_throughput_samples_per_second": 0.10,
        "incremental_peak_rss_mib": 0.05,
    },
    "efficiency_priority": {
        "pooled_wer": 0.15,
        "exact_sentence_accuracy": 0.15,
        "trainable_parameters": 0.25,
        "batch1_latency_median_ms": 0.20,
        "batch16_throughput_samples_per_second": 0.15,
        "incremental_peak_rss_mib": 0.10,
    },
}


def quality_eligible(row: Mapping[str, object]) -> bool:
    return (
        float(row["pooled_wer"]) <= QUALITY_THRESHOLDS["maximum_pooled_wer"]
        and float(row["exact_sentence_accuracy"])
        >= QUALITY_THRESHOLDS["minimum_exact_sentence_accuracy"]
    )


def lightweight_eligible(row: Mapping[str, object]) -> bool:
    return int(row["trainable_parameters"]) <= LIGHTWEIGHT_PARAMETER_LIMIT


def normalize_utility(
    values: Sequence[float],
    *,
    direction: str,
) -> list[float]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("Cannot normalize an empty sequence.")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("Direction must be minimize or maximize.")
    minimum = min(numeric)
    maximum = max(numeric)
    if maximum == minimum:
        return [1.0] * len(numeric)
    if direction == "maximize":
        return [(value - minimum) / (maximum - minimum) for value in numeric]
    return [(maximum - value) / (maximum - minimum) for value in numeric]


def weighted_scores(
    rows: Sequence[Mapping[str, object]],
    weights: Mapping[str, float],
) -> dict[str, float]:
    if not rows:
        raise ValueError("At least one eligible model is required.")
    if set(weights) != set(CRITERIA_DIRECTIONS):
        raise ValueError("Weight criteria do not match the decision criteria.")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-12:
        raise ValueError("Scenario weights must sum to one.")

    active = tuple(
        criterion
        for criterion in CRITERIA_DIRECTIONS
        if all(row.get(criterion) is not None for row in rows)
    )
    if not active:
        raise ValueError("No decision criterion is available.")
    active_weight = sum(float(weights[criterion]) for criterion in active)
    if active_weight <= 0.0:
        raise ValueError("Available criteria have zero total weight.")

    utilities: dict[str, list[float]] = {}
    for criterion in active:
        direction = CRITERIA_DIRECTIONS[criterion]
        utilities[criterion] = normalize_utility(
            [float(row[criterion]) for row in rows], direction=direction
        )
    return {
        str(row["model_id"]): sum(
            (float(weights[criterion]) / active_weight)
            * utilities[criterion][index]
            for criterion in active
        )
        for index, row in enumerate(rows)
    }


def pareto_frontier(rows: Sequence[Mapping[str, object]]) -> set[str]:
    """Return nondominated models across all available stated criteria."""
    active = tuple(
        criterion
        for criterion in CRITERIA_DIRECTIONS
        if all(row.get(criterion) is not None for row in rows)
    )
    if not active:
        raise ValueError("No Pareto criterion is available.")
    frontier: set[str] = set()
    for candidate in rows:
        dominated = False
        for challenger in rows:
            if challenger is candidate:
                continue
            no_worse: list[bool] = []
            strictly_better: list[bool] = []
            for criterion in active:
                direction = CRITERIA_DIRECTIONS[criterion]
                candidate_value = float(candidate[criterion])
                challenger_value = float(challenger[criterion])
                if direction == "minimize":
                    no_worse.append(challenger_value <= candidate_value)
                    strictly_better.append(challenger_value < candidate_value)
                else:
                    no_worse.append(challenger_value >= candidate_value)
                    strictly_better.append(challenger_value > candidate_value)
            if all(no_worse) and any(strictly_better):
                dominated = True
                break
        if not dominated:
            frontier.add(str(candidate["model_id"]))
    return frontier


def _numeric(row: Mapping[str, str], key: str) -> float | None:
    value = row[key]
    if value == "" and key == "incremental_peak_rss_mib":
        return None
    return float(value)


def load_decision_rows(project_root: str | Path) -> list[dict[str, object]]:
    root = Path(project_root).resolve()
    output_root = root / "artifacts" / "analysis" / "stage34"
    performance = {
        row["model_id"]: row
        for row in read_csv(output_root / "stage34b_model_metrics.csv")
    }
    intervals = {
        row["model_id"]: row
        for row in read_csv(output_root / "stage34c_model_confidence_intervals.csv")
    }
    efficiency = {
        row["model_id"]: row
        for row in read_csv(output_root / "stage34d_efficiency_metrics.csv")
    }
    if set(performance) != set(MODEL_ORDER) or set(intervals) != set(MODEL_ORDER) or set(efficiency) != set(MODEL_ORDER):
        raise RuntimeError("Stage 34 model sets do not match.")

    rows: list[dict[str, object]] = []
    for model_id in MODEL_ORDER:
        p = performance[model_id]
        ci = intervals[model_id]
        e = efficiency[model_id]
        row: dict[str, object] = {
            "model_id": model_id,
            "pooled_wer": _numeric(p, "pooled_wer"),
            "pooled_wer_ci95_lower": _numeric(ci, "pooled_wer_ci95_lower"),
            "pooled_wer_ci95_upper": _numeric(ci, "pooled_wer_ci95_upper"),
            "exact_sentence_accuracy": _numeric(p, "exact_sentence_accuracy"),
            "exact_accuracy_ci95_lower": _numeric(ci, "exact_accuracy_ci95_lower"),
            "exact_accuracy_ci95_upper": _numeric(ci, "exact_accuracy_ci95_upper"),
            "ser": _numeric(p, "ser"),
            "trainable_parameters": int(e["trainable_parameters"]),
            "serialized_state_dict_mib": _numeric(e, "serialized_state_dict_mib"),
            "batch1_latency_median_ms": _numeric(e, "batch1_latency_median_ms"),
            "batch16_throughput_samples_per_second": _numeric(
                e, "batch16_throughput_samples_per_second"
            ),
            "incremental_peak_rss_mib": _numeric(e, "incremental_peak_rss_mib"),
        }
        row["quality_eligible"] = quality_eligible(row)
        row["lightweight_eligible"] = lightweight_eligible(row)
        rows.append(row)
    return rows


def statistical_wer_tie_groups(project_root: str | Path) -> list[list[str]]:
    """Build connected groups whose paired WER differences were unresolved."""
    root = Path(project_root).resolve()
    pairs = read_csv(
        root
        / "artifacts"
        / "analysis"
        / "stage34"
        / "stage34c_pairwise_tests.csv"
    )
    adjacency = {model_id: set() for model_id in MODEL_ORDER}
    for row in pairs:
        if row["wer_difference_direction"] == "not_resolved":
            left = row["model_a"]
            right = row["model_b"]
            adjacency[left].add(right)
            adjacency[right].add(left)

    groups: list[list[str]] = []
    visited: set[str] = set()
    for model_id in MODEL_ORDER:
        if model_id in visited or not adjacency[model_id]:
            continue
        stack = [model_id]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        visited.update(component)
        groups.append(sorted(component, key=MODEL_ORDER.index))
    return groups


def build_decision_tables(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    output_root = root / "artifacts" / "analysis" / "stage34"
    rows = load_decision_rows(root)
    eligible = [row for row in rows if bool(row["quality_eligible"])]
    if not eligible:
        raise RuntimeError("No model meets the declared quality threshold.")
    frontier = pareto_frontier(eligible)
    active_criteria = [
        criterion
        for criterion in CRITERIA_DIRECTIONS
        if all(row.get(criterion) is not None for row in eligible)
    ]

    tie_groups = statistical_wer_tie_groups(root)
    scoring_rows = [dict(row) for row in eligible]
    scoring_by_model = {str(row["model_id"]): row for row in scoring_rows}
    for group in tie_groups:
        eligible_group = [model for model in group if model in scoring_by_model]
        if len(eligible_group) < 2:
            continue
        common_wer = statistics.fmean(
            float(scoring_by_model[model]["pooled_wer"])
            for model in eligible_group
        )
        for model in eligible_group:
            scoring_by_model[model]["pooled_wer"] = common_wer

    scenario_scores = {
        name: weighted_scores(scoring_rows, weights)
        for name, weights in WEIGHT_SCENARIOS.items()
    }
    scenario_rows: list[dict[str, object]] = []
    for scenario, scores in scenario_scores.items():
        order = sorted(scores, key=lambda model: (-scores[model], MODEL_ORDER.index(model)))
        for rank, model_id in enumerate(order, start=1):
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "rank": rank,
                    "model_id": model_id,
                    "score": scores[model_id],
                    "scenario_winner": rank == 1,
                }
            )

    average_scores = {
        str(row["model_id"]): statistics.fmean(
            scores[str(row["model_id"])] for scores in scenario_scores.values()
        )
        for row in eligible
    }
    lightweight_candidates = [
        row for row in eligible if bool(row["lightweight_eligible"])
    ]
    if not lightweight_candidates:
        raise RuntimeError("No quality-eligible lightweight model exists.")
    recommended_lightweight = max(
        (str(row["model_id"]) for row in lightweight_candidates),
        key=lambda model: (average_scores[model], -int(next(row["trainable_parameters"] for row in rows if row["model_id"] == model))),
    )
    accuracy_reference = min(
        eligible,
        key=lambda row: (float(row["pooled_wer"]), -float(row["exact_sentence_accuracy"])),
    )["model_id"]

    for row in rows:
        model_id = str(row["model_id"])
        row["pareto_frontier"] = model_id in frontier
        row["mean_scenario_score"] = average_scores.get(model_id, 0.0)
        row["mean_scenario_rank"] = (
            1
            + sum(
                average_scores[other] > average_scores[model_id]
                for other in average_scores
            )
            if model_id in average_scores
            else "not_eligible"
        )
        row["final_role"] = (
            "accuracy_reference"
            if model_id == accuracy_reference
            else "recommended_lightweight"
            if model_id == recommended_lightweight
            else "eligible_comparator"
            if bool(row["quality_eligible"])
            else "performance_threshold_not_met"
        )

    matrix_path = output_root / "stage34e_decision_matrix.csv"
    scenarios_path = output_root / "stage34e_sensitivity_ranking.csv"
    write_csv(matrix_path, rows)
    write_csv(scenarios_path, scenario_rows)
    return {
        "rows": rows,
        "scenario_rows": scenario_rows,
        "recommended_lightweight": recommended_lightweight,
        "accuracy_reference": accuracy_reference,
        "pareto_frontier": sorted(frontier, key=MODEL_ORDER.index),
        "statistical_wer_tie_groups": tie_groups,
        "active_criteria": active_criteria,
        "memory_criterion_used": "incremental_peak_rss_mib" in active_criteria,
        "matrix_path": matrix_path,
        "scenarios_path": scenarios_path,
    }


def report_markdown(result: Mapping[str, object]) -> str:
    rows = result["rows"]
    scenario_rows = result["scenario_rows"]
    lines = [
        "# Stage 34E — Final Multi-Criteria Model Selection",
        "",
        "## Decision rule",
        "",
        "Models must first satisfy WER ≤ 0.10 and exact-sentence accuracy ≥ 0.75.",
        "Eligible models are then compared under quality-priority, balanced and",
        "efficiency-priority weight scenarios. Scores use min-max utility within",
        "the eligible set. The recommendation is the eligible sub-one-million-",
        "parameter model with the highest mean score across all three scenarios.",
        "This is a transparent decision aid, not a claim that weights are universal.",
        "If the operating system cannot expose peak RSS, the memory criterion is",
        "marked unavailable and its weight is proportionally redistributed across",
        "the remaining criteria; the benchmark still reports this limitation.",
        "Models with an unresolved paired WER difference receive their common group",
        "mean WER during scoring, so an insignificant numerical difference cannot",
        "change the multi-criteria decision.",
        "",
        "## Combined evidence",
        "",
        "| Model | WER | Exact accuracy | Parameters | Batch-1 ms | Batch-16 samples/s | Quality eligible | Pareto | Role |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {wer:.6f} | {accuracy:.6f} | {parameters:,} | {latency:.3f} | "
            "{throughput:.2f} | {eligible} | {pareto} | {role} |".format(
                model=str(row["model_id"]).upper(),
                wer=float(row["pooled_wer"]),
                accuracy=float(row["exact_sentence_accuracy"]),
                parameters=int(row["trainable_parameters"]),
                latency=float(row["batch1_latency_median_ms"]),
                throughput=float(row["batch16_throughput_samples_per_second"]),
                eligible=row["quality_eligible"],
                pareto=row["pareto_frontier"],
                role=row["final_role"],
            )
        )
    lines.extend(
        [
            "",
            "## Weight sensitivity",
            "",
            "| Scenario | Rank | Model | Score |",
            "|---|---:|---|---:|",
        ]
    )
    for row in scenario_rows:
        lines.append(
            f"| {row['scenario']} | {row['rank']} | {str(row['model_id']).upper()} | "
            f"{float(row['score']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Final interpretation",
            "",
            f"- Accuracy reference: {str(result['accuracy_reference']).upper()}.",
            f"- Recommended lightweight model: {str(result['recommended_lightweight']).upper()}.",
            "- The recommendation combines frozen recognition, uncertainty and",
            "  same-machine efficiency evidence; it is not based on accuracy alone.",
            "- C2 and C4 must not be separated by WER significance because Stage",
            "  34C did not resolve their WER difference at 95% confidence.",
            "",
        ]
    )
    return "\n".join(lines)
