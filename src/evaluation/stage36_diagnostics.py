"""Stage 36 fold-level error diagnostics and dependency-free SVG figures."""

from __future__ import annotations

import html
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.stage35_evidence import MODEL_ORDER


COLORS = {
    "g1": "#173F5F",
    "c1": "#6C757D",
    "c2": "#00796B",
    "c3": "#C62828",
    "c4": "#EF6C00",
}
STAGE34_ROOT = Path("artifacts/analysis/stage34")
STAGE35_ROOT = Path("artifacts/analysis/stage35")
STAGE36_ROOT = Path("artifacts/analysis/stage36")


def _first(row: Mapping[str, Any], names: Iterable[str], *, required: bool = True) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    if required:
        raise KeyError(f"Missing required field; accepted aliases: {tuple(names)}")
    return None


def _number(row: Mapping[str, Any], *names: str, required: bool = True) -> float | None:
    value = _first(row, names, required=required)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {names}: {value}")
    return number


def _integer(row: Mapping[str, Any], *names: str, required: bool = True) -> int | None:
    value = _number(row, *names, required=required)
    return None if value is None else int(value)


def normalize_fold_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and rank the 25 frozen model-fold rows."""
    normalized: list[dict[str, Any]] = []
    for source in rows:
        model_id = str(_first(source, ("model_id", "model"))).lower()
        if model_id not in MODEL_ORDER:
            raise ValueError(f"Unexpected model id: {model_id}")
        fold = _integer(source, "outer_fold", "fold", "fold_id")
        substitutions = _integer(
            source, "substitutions", "substitutions_total", required=False
        )
        deletions = _integer(source, "deletions", "deletions_total", required=False)
        insertions = _integer(source, "insertions", "insertions_total", required=False)
        available = all(
            value is not None for value in (substitutions, deletions, insertions)
        )
        total = (
            int(substitutions) + int(deletions) + int(insertions)
            if available
            else None
        )
        normalized.append(
            {
                "model_id": model_id,
                "outer_fold": fold,
                "pooled_wer": _number(source, "pooled_wer", "wer"),
                "exact_sentence_accuracy": _number(
                    source,
                    "exact_sentence_accuracy",
                    "sentence_accuracy",
                    "accuracy",
                ),
                "ser": _number(source, "ser", "sentence_error_rate"),
                "substitutions": substitutions if substitutions is not None else "",
                "deletions": deletions if deletions is not None else "",
                "insertions": insertions if insertions is not None else "",
                "total_edits": total if total is not None else "",
                "substitution_share": substitutions / total if total else "",
                "deletion_share": deletions / total if total else "",
                "insertion_share": insertions / total if total else "",
                "error_counts_available": available,
            }
        )

    expected = {
        (model_id, fold)
        for fold in range(1, 6)
        for model_id in MODEL_ORDER
    }
    observed = {
        (str(row["model_id"]), int(row["outer_fold"]))
        for row in normalized
    }
    if observed != expected or len(normalized) != 25:
        raise RuntimeError(
            "Fold metrics must contain one row for every model and outer fold."
        )

    for fold in range(1, 6):
        fold_rows = [row for row in normalized if int(row["outer_fold"]) == fold]
        ordered = sorted(
            fold_rows,
            key=lambda row: (
                float(row["pooled_wer"]),
                -float(row["exact_sentence_accuracy"]),
                MODEL_ORDER.index(str(row["model_id"])),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row["wer_rank_in_fold"] = rank
    return sorted(
        normalized,
        key=lambda row: (
            int(row["outer_fold"]),
            MODEL_ORDER.index(str(row["model_id"])),
        ),
    )


def build_stability_rows(
    fold_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        rows = sorted(
            [row for row in fold_rows if row["model_id"] == model_id],
            key=lambda row: int(row["outer_fold"]),
        )
        if len(rows) != 5:
            raise RuntimeError(f"Expected five folds for {model_id}.")
        wers = [float(row["pooled_wer"]) for row in rows]
        accuracies = [float(row["exact_sentence_accuracy"]) for row in rows]
        sers = [float(row["ser"]) for row in rows]
        ranks = [int(row["wer_rank_in_fold"]) for row in rows]
        mean_wer = statistics.fmean(wers)
        best = min(range(5), key=wers.__getitem__)
        worst = max(range(5), key=wers.__getitem__)
        result.append(
            {
                "model_id": model_id,
                "folds": 5,
                "pooled_wer_mean": mean_wer,
                "pooled_wer_sample_sd": statistics.stdev(wers),
                "pooled_wer_min": min(wers),
                "pooled_wer_max": max(wers),
                "pooled_wer_range": max(wers) - min(wers),
                "pooled_wer_coefficient_of_variation": (
                    statistics.stdev(wers) / mean_wer if mean_wer else 0.0
                ),
                "exact_accuracy_mean": statistics.fmean(accuracies),
                "exact_accuracy_sample_sd": statistics.stdev(accuracies),
                "ser_mean": statistics.fmean(sers),
                "ser_sample_sd": statistics.stdev(sers),
                "best_wer_fold": int(rows[best]["outer_fold"]),
                "worst_wer_fold": int(rows[worst]["outer_fold"]),
                "mean_wer_rank": statistics.fmean(ranks),
                "fold_wins": sum(rank == 1 for rank in ranks),
            }
        )
    return result


def build_error_rows(
    fold_rows: Sequence[Mapping[str, Any]],
    fallback_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    fallback = {
        str(row["model_id"]).lower(): row
        for row in fallback_rows
    }
    for model_id in MODEL_ORDER:
        rows = [row for row in fold_rows if row["model_id"] == model_id]
        available = all(bool(row["error_counts_available"]) for row in rows)
        if available:
            substitutions = sum(int(row["substitutions"]) for row in rows)
            deletions = sum(int(row["deletions"]) for row in rows)
            insertions = sum(int(row["insertions"]) for row in rows)
            source = "stage34b_fold_metrics"
        elif model_id in fallback:
            source_row = fallback[model_id]
            substitutions = _integer(
                source_row, "substitutions", required=False
            )
            deletions = _integer(source_row, "deletions", required=False)
            insertions = _integer(source_row, "insertions", required=False)
            available = all(
                value is not None
                for value in (substitutions, deletions, insertions)
            )
            source = "stage35_aggregate_error_profile"
        else:
            substitutions = deletions = insertions = None
            source = "not_available"
        if available:
            total = substitutions + deletions + insertions
            counts = {
                "substitution": substitutions,
                "deletion": deletions,
                "insertion": insertions,
            }
            dominant = max(counts, key=counts.get) if total else "none"
        else:
            substitutions = deletions = insertions = total = None
            dominant = "not_available"
            source = "not_available"
        result.append(
            {
                "model_id": model_id,
                "substitutions": substitutions if substitutions is not None else "",
                "deletions": deletions if deletions is not None else "",
                "insertions": insertions if insertions is not None else "",
                "total_edits": total if total is not None else "",
                "substitution_share": substitutions / total if total else "",
                "deletion_share": deletions / total if total else "",
                "insertion_share": insertions / total if total else "",
                "dominant_error_type": dominant,
                "fold_error_counts_available": available,
                "error_count_source": source,
            }
        )
    return result


def c2_c4_fold_comparison(
    fold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (str(row["model_id"]), int(row["outer_fold"])): row
        for row in fold_rows
    }
    c2_wins = c4_wins = ties = 0
    deltas: list[float] = []
    for fold in range(1, 6):
        c2 = float(by_key[("c2", fold)]["pooled_wer"])
        c4 = float(by_key[("c4", fold)]["pooled_wer"])
        deltas.append(c2 - c4)
        if c2 < c4:
            c2_wins += 1
        elif c4 < c2:
            c4_wins += 1
        else:
            ties += 1
    return {
        "c2_lower_wer_folds": c2_wins,
        "c4_lower_wer_folds": c4_wins,
        "equal_wer_folds": ties,
        "mean_fold_delta_c2_minus_c4": statistics.fmean(deltas),
        "statistical_conclusion": "not_resolved_at_95_percent",
    }


def _svg_text(x: float, y: float, value: str, **attrs: Any) -> str:
    rendered = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(item))}"'
        for key, item in attrs.items()
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" {rendered}>'
        f"{html.escape(value)}</text>"
    )


def fold_wer_svg(fold_rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 1100, 660
    left, right, top, bottom = 95, 35, 75, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(row["pooled_wer"]) for row in fold_rows)
    y_max = max(0.10, math.ceil(maximum * 10) / 10)

    def x_position(fold: int) -> float:
        return left + (fold - 1) * plot_width / 4

    def y_position(value: float) -> float:
        return top + plot_height * (1.0 - value / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        _svg_text(
            55,
            38,
            "Five-fold pooled WER by model",
            font_size=25,
            font_weight="700",
            fill="#172B4D",
        ),
        _svg_text(
            55,
            63,
            "Lower is better; points are frozen outer-fold results",
            font_size=15,
            fill="#52606D",
        ),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            f'y2="{y:.1f}" stroke="#D9E2EC" stroke-width="1"/>'
        )
        parts.append(
            _svg_text(
                left - 12,
                y + 5,
                f"{value:.2f}",
                text_anchor="end",
                font_size=13,
                fill="#52606D",
            )
        )
    for fold in range(1, 6):
        x = x_position(fold)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{height-bottom}" stroke="#EEF2F6" stroke-width="1"/>'
        )
        parts.append(
            _svg_text(
                x,
                height - 55,
                f"Fold {fold}",
                text_anchor="middle",
                font_size=14,
                fill="#334E68",
            )
        )
    for model_id in MODEL_ORDER:
        rows = sorted(
            [row for row in fold_rows if row["model_id"] == model_id],
            key=lambda row: int(row["outer_fold"]),
        )
        points = " ".join(
            f'{x_position(int(row["outer_fold"])):.1f},'
            f'{y_position(float(row["pooled_wer"])):.1f}'
            for row in rows
        )
        color = COLORS[model_id]
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3"/>'
        )
        for row in rows:
            parts.append(
                f'<circle cx="{x_position(int(row["outer_fold"])):.1f}" '
                f'cy="{y_position(float(row["pooled_wer"])):.1f}" r="5" '
                f'fill="{color}" stroke="#FFFFFF" stroke-width="2"/>'
            )
    legend_y = height - 18
    for index, model_id in enumerate(MODEL_ORDER):
        x = 240 + index * 135
        parts.append(
            f'<line x1="{x}" y1="{legend_y-5}" x2="{x+24}" '
            f'y2="{legend_y-5}" stroke="{COLORS[model_id]}" stroke-width="4"/>'
        )
        parts.append(
            _svg_text(
                x + 31,
                legend_y,
                model_id.upper(),
                font_size=14,
                font_weight="700",
                fill="#243B53",
            )
        )
    centre = top + plot_height / 2
    parts.append(
        _svg_text(
            25,
            centre,
            "Pooled WER",
            font_size=15,
            fill="#334E68",
            transform=f"rotate(-90 25 {centre:.1f})",
            text_anchor="middle",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def quality_efficiency_svg(
    publication_rows: Sequence[Mapping[str, Any]],
) -> str:
    width, height = 1100, 660
    left, right, top, bottom = 105, 45, 80, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    parameters = [float(row["trainable_parameters"]) for row in publication_rows]
    wers = [float(row["pooled_wer"]) for row in publication_rows]
    log_min = math.log10(min(parameters)) - 0.08
    log_max = math.log10(max(parameters)) + 0.08
    y_max = max(0.10, math.ceil(max(wers) * 10) / 10)

    def x_position(value: float) -> float:
        return left + (
            (math.log10(value) - log_min) / (log_max - log_min) * plot_width
        )

    def y_position(value: float) -> float:
        return top + plot_height * value / y_max

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        _svg_text(
            55,
            38,
            "Recognition quality versus model size",
            font_size=25,
            font_weight="700",
            fill="#172B4D",
        ),
        _svg_text(
            55,
            63,
            "Pooled WER versus trainable parameters; lower-left is preferred",
            font_size=15,
            fill="#52606D",
        ),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" '
            f'y2="{y:.1f}" stroke="#D9E2EC" stroke-width="1"/>'
        )
        parts.append(
            _svg_text(
                left - 12,
                y + 5,
                f"{value:.2f}",
                text_anchor="end",
                font_size=13,
                fill="#52606D",
            )
        )
    for value, label in (
        (250_000, "0.25M"),
        (500_000, "0.5M"),
        (1_000_000, "1M"),
        (2_000_000, "2M"),
        (5_000_000, "5M"),
    ):
        if log_min <= math.log10(value) <= log_max:
            x = x_position(value)
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                f'y2="{height-bottom}" stroke="#EEF2F6" stroke-width="1"/>'
            )
            parts.append(
                _svg_text(
                    x,
                    height - 62,
                    label,
                    text_anchor="middle",
                    font_size=13,
                    fill="#52606D",
                )
            )
    for row in publication_rows:
        model_id = str(row["model_id"])
        x = x_position(float(row["trainable_parameters"]))
        y = y_position(float(row["pooled_wer"]))
        radius = 12 if model_id == "c2" else 9
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
            f'fill="{COLORS[model_id]}" stroke="#FFFFFF" stroke-width="3"/>'
        )
        label_y = y - 16 if model_id != "g1" else y + 30
        parts.append(
            _svg_text(
                x + 14,
                label_y,
                model_id.upper(),
                font_size=16,
                font_weight="700",
                fill=COLORS[model_id],
            )
        )
    parts.append(
        _svg_text(
            left + plot_width / 2,
            height - 20,
            "Trainable parameters (log scale)",
            text_anchor="middle",
            font_size=15,
            fill="#334E68",
        )
    )
    centre = top + plot_height / 2
    parts.append(
        _svg_text(
            25,
            centre,
            "Pooled WER",
            font_size=15,
            fill="#334E68",
            transform=f"rotate(-90 25 {centre:.1f})",
            text_anchor="middle",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_report(
    stability_rows: Sequence[Mapping[str, Any]],
    error_rows: Sequence[Mapping[str, Any]],
    c2_c4: Mapping[str, Any],
) -> str:
    stability = {str(row["model_id"]): row for row in stability_rows}
    errors = {str(row["model_id"]): row for row in error_rows}
    lines = [
        "# Stage 36 — Fold-Level Error Analysis and Thesis Figures",
        "",
        "## Purpose",
        "",
        "This stage diagnoses variation and failure modes in the 25 frozen",
        "mixed-subject five-fold evaluations. It does not train or rescore a model.",
        "",
        "## Fold stability",
        "",
        "| Model | Mean WER | SD | Range | Best fold | Worst fold | Fold wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        row = stability[model_id]
        lines.append(
            f"| {model_id.upper()} | {float(row['pooled_wer_mean']):.6f} | "
            f"{float(row['pooled_wer_sample_sd']):.6f} | "
            f"{float(row['pooled_wer_range']):.6f} | "
            f"{row['best_wer_fold']} | {row['worst_wer_fold']} | "
            f"{row['fold_wins']} |"
        )
    lines.extend(
        [
            "",
            "## Error composition",
            "",
            "| Model | Substitutions | Deletions | Insertions | Dominant type |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for model_id in MODEL_ORDER:
        row = errors[model_id]
        lines.append(
            f"| {model_id.upper()} | {row['substitutions']} | "
            f"{row['deletions']} | {row['insertions']} | "
            f"{row['dominant_error_type']} |"
        )
    lines.extend(
        [
            "",
            "## C2 and C4",
            "",
            f"C2 obtained lower fold WER in {c2_c4['c2_lower_wer_folds']} folds; "
            f"C4 obtained lower fold WER in {c2_c4['c4_lower_wer_folds']} folds. "
            "The paired Stage 34 inference remains authoritative: the overall WER "
            "difference is not resolved at 95% confidence.",
            "",
            "## Thesis interpretation",
            "",
            "- G1 is the stable recognition-quality reference, not the lightweight recommendation.",
            "- C2 is the selected lightweight quality–efficiency compromise.",
            "- C2 and C4 exchange fold-level advantages; this supports cautious statistical wording.",
            "- Deletion-heavy outputs indicate CTC under-generation rather than simple class confusion.",
            "- C3's low parameter count does not compensate for its recognition failure.",
            "",
            "## Figures",
            "",
            "- stage36_fold_wer.svg: cross-fold WER variation.",
            "- stage36_quality_efficiency.svg: WER versus trainable parameters.",
            "",
            "Both SVG files are vector graphics suitable for the thesis document.",
            "",
        ]
    )
    return "\n".join(lines)
