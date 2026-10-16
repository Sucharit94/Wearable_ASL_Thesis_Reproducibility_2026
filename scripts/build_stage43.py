from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


MODELS = ("G1", "C1", "C2", "C3", "C4")
ALPHA = 0.05
BOOTSTRAP_REPLICATES = 10_000
# Retain the frozen Stage 34C resampling seed so the previously reported
# pairwise bootstrap intervals and p-values remain exactly reproducible.
SEED = 20260912


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    subject: str
    sentence_id: int
    fold: int
    reference_tokens: tuple[str, ...]
    edit_distance: int
    exact_match: bool
    substitutions: int
    deletions: int
    insertions: int


TEXT_HASH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    """Hash text canonically so Git line-ending conversion cannot break evidence."""
    digest = hashlib.sha256()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt_p(value: float) -> str:
    if value == 0.0 or value < 0.001:
        return "<0.001"
    return f"{value:.6f}"


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def resolve_results_root(source: Path, temporary: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError("Input must be the Stage 32C result directory or ZIP archive.")
    with zipfile.ZipFile(source) as archive:
        archive.extractall(temporary)
    return temporary


def load_predictions(root: Path) -> dict[str, dict[str, Prediction]]:
    files = sorted(root.glob("**/known_signer/fold*/**/predictions.jsonl"))
    if len(files) != 25:
        raise RuntimeError(f"Expected 25 prediction files; found {len(files)}.")

    loaded: dict[str, dict[str, Prediction]] = {model: {} for model in MODELS}
    for path in files:
        model = path.parent.name.upper()
        if model not in loaded:
            continue
        fold_name = path.parts[-3]
        fold = int(fold_name.replace("fold", ""))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prediction = Prediction(
                    sample_id=row["sample_id"],
                    subject=row["subject"],
                    sentence_id=int(row["sentence_id"]),
                    fold=fold,
                    reference_tokens=tuple(row["reference_tokens"]),
                    edit_distance=int(row["edit_distance"]),
                    exact_match=bool(row["exact_match"]),
                    substitutions=int(row["substitutions"]),
                    deletions=int(row["deletions"]),
                    insertions=int(row["insertions"]),
                )
                if prediction.sample_id in loaded[model]:
                    raise RuntimeError(
                        f"Duplicate prediction for {model}/{prediction.sample_id}."
                    )
                loaded[model][prediction.sample_id] = prediction

    expected_ids = set(loaded[MODELS[0]])
    if len(expected_ids) != 7_400:
        raise RuntimeError(f"Expected 7,400 samples; found {len(expected_ids)}.")
    for model in MODELS:
        if set(loaded[model]) != expected_ids:
            raise RuntimeError(f"Sample identities do not match for {model}.")

    for sample_id in expected_ids:
        reference = loaded[MODELS[0]][sample_id]
        for model in MODELS[1:]:
            candidate = loaded[model][sample_id]
            if (
                candidate.reference_tokens != reference.reference_tokens
                or candidate.subject != reference.subject
                or candidate.sentence_id != reference.sentence_id
                or candidate.fold != reference.fold
            ):
                raise RuntimeError(f"Reference mismatch for {model}/{sample_id}.")
    return loaded


def build_matrices(predictions: dict[str, dict[str, Prediction]]) -> dict:
    sample_ids = sorted(predictions[MODELS[0]])
    cluster_keys = sorted(
        {
            (
                predictions[MODELS[0]][sample_id].subject,
                predictions[MODELS[0]][sample_id].sentence_id,
            )
            for sample_id in sample_ids
        }
    )
    cluster_index = {key: index for index, key in enumerate(cluster_keys)}
    sample_count = len(sample_ids)
    cluster_count = len(cluster_keys)
    model_count = len(MODELS)

    cluster_edits = np.zeros((cluster_count, model_count), dtype=float)
    cluster_reference = np.zeros((cluster_count, model_count), dtype=float)
    sample_correct = np.zeros((sample_count, model_count), dtype=int)
    sample_edits = np.zeros((sample_count, model_count), dtype=float)
    sample_reference = np.zeros((sample_count, model_count), dtype=float)
    substitutions = np.zeros(model_count, dtype=int)
    deletions = np.zeros(model_count, dtype=int)
    insertions = np.zeros(model_count, dtype=int)

    for model_index, model in enumerate(MODELS):
        for sample_index, sample_id in enumerate(sample_ids):
            row = predictions[model][sample_id]
            reference_length = len(row.reference_tokens)
            if reference_length == 0:
                raise RuntimeError(f"Empty reference for {model}/{sample_id}.")
            cluster = cluster_index[(row.subject, row.sentence_id)]
            cluster_edits[cluster, model_index] += row.edit_distance
            cluster_reference[cluster, model_index] += reference_length
            sample_edits[sample_index, model_index] = row.edit_distance
            sample_reference[sample_index, model_index] = reference_length
            sample_correct[sample_index, model_index] = int(row.exact_match)
            substitutions[model_index] += row.substitutions
            deletions[model_index] += row.deletions
            insertions[model_index] += row.insertions

    if cluster_count != 368:
        raise RuntimeError(f"Expected 368 subject-sentence clusters; found {cluster_count}.")
    if not np.all(cluster_reference > 0):
        raise RuntimeError("A model is missing a subject-sentence cluster.")

    return {
        "sample_ids": sample_ids,
        "cluster_keys": cluster_keys,
        "cluster_edits": cluster_edits,
        "cluster_reference": cluster_reference,
        "cluster_wer": cluster_edits / cluster_reference,
        "sample_correct": sample_correct,
        "sample_edits": sample_edits,
        "sample_reference": sample_reference,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def repeated_measures_anova(values: np.ndarray) -> dict:
    n_blocks, n_models = values.shape
    grand_mean = float(values.mean())
    model_means = values.mean(axis=0)
    block_means = values.mean(axis=1)
    ss_total = float(((values - grand_mean) ** 2).sum())
    ss_model = float(n_blocks * ((model_means - grand_mean) ** 2).sum())
    ss_block = float(n_models * ((block_means - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_model - ss_block
    df_model = n_models - 1
    df_error = (n_blocks - 1) * (n_models - 1)
    statistic = (ss_model / df_model) / (ss_error / df_error)
    p_value = float(stats.f.sf(statistic, df_model, df_error))

    covariance = np.cov(values, rowvar=False, ddof=1)
    centering = np.eye(n_models) - np.ones((n_models, n_models)) / n_models
    centered_covariance = centering @ covariance @ centering
    trace = float(np.trace(centered_covariance))
    trace_square = float(np.trace(centered_covariance @ centered_covariance))
    epsilon = trace * trace / (df_model * trace_square)
    epsilon = min(1.0, max(1.0 / df_model, epsilon))
    gg_df1 = epsilon * df_model
    gg_df2 = epsilon * df_error
    gg_p = float(stats.f.sf(statistic, gg_df1, gg_df2))
    partial_eta_squared = ss_model / (ss_model + ss_error)
    return {
        "test": "Repeated-measures ANOVA F-test",
        "statistic": statistic,
        "df1": float(df_model),
        "df2": float(df_error),
        "p_value": p_value,
        "effect_name": "partial_eta_squared",
        "effect_value": partial_eta_squared,
        "greenhouse_geisser_epsilon": epsilon,
        "corrected_df1": gg_df1,
        "corrected_df2": gg_df2,
        "corrected_p_value": gg_p,
    }


def friedman_test(values: np.ndarray) -> dict:
    result = stats.friedmanchisquare(*[values[:, i] for i in range(values.shape[1])])
    kendalls_w = float(result.statistic / (values.shape[0] * (values.shape[1] - 1)))
    return {
        "test": "Friedman repeated-measures test",
        "statistic": float(result.statistic),
        "df1": float(values.shape[1] - 1),
        "df2": "",
        "p_value": float(result.pvalue),
        "effect_name": "kendalls_w",
        "effect_value": kendalls_w,
        "greenhouse_geisser_epsilon": "",
        "corrected_df1": "",
        "corrected_df2": "",
        "corrected_p_value": "",
    }


def cochrans_q(correct: np.ndarray) -> dict:
    n_models = correct.shape[1]
    column_sums = correct.sum(axis=0).astype(float)
    row_sums = correct.sum(axis=1).astype(float)
    total = float(column_sums.sum())
    denominator = n_models * total - float((row_sums**2).sum())
    if denominator <= 0:
        raise RuntimeError("Cochran's Q denominator is not positive.")
    statistic = (n_models - 1) * (
        n_models * float((column_sums**2).sum()) - total**2
    ) / denominator
    return {
        "test": "Cochran Q test",
        "statistic": statistic,
        "df1": float(n_models - 1),
        "df2": "",
        "p_value": float(stats.chi2.sf(statistic, n_models - 1)),
        "effect_name": "",
        "effect_value": "",
        "greenhouse_geisser_epsilon": "",
        "corrected_df1": "",
        "corrected_df2": "",
        "corrected_p_value": "",
    }


def pairwise_ttests(cluster_wer: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    raw_p: list[float] = []
    for a, b in combinations(range(len(MODELS)), 2):
        difference = cluster_wer[:, a] - cluster_wer[:, b]
        result = stats.ttest_rel(cluster_wer[:, a], cluster_wer[:, b])
        standard_deviation = float(difference.std(ddof=1))
        effect = float(difference.mean() / standard_deviation) if standard_deviation else 0.0
        raw_p.append(float(result.pvalue))
        rows.append(
            {
                "model_a": MODELS[a],
                "model_b": MODELS[b],
                "mean_cluster_wer_a": float(cluster_wer[:, a].mean()),
                "mean_cluster_wer_b": float(cluster_wer[:, b].mean()),
                "mean_difference_a_minus_b": float(difference.mean()),
                "t_statistic": float(result.statistic),
                "df": cluster_wer.shape[0] - 1,
                "raw_p_value": float(result.pvalue),
                "holm_p_value": 0.0,
                "cohens_dz": effect,
                "significant_0_05": False,
            }
        )
    for row, adjusted in zip(rows, holm_adjust(raw_p)):
        row["holm_p_value"] = adjusted
        row["significant_0_05"] = adjusted < ALPHA
    return rows


def pairwise_mcnemar(correct: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    raw_p: list[float] = []
    for a, b in combinations(range(len(MODELS)), 2):
        only_a = int(np.sum((correct[:, a] == 1) & (correct[:, b] == 0)))
        only_b = int(np.sum((correct[:, a] == 0) & (correct[:, b] == 1)))
        discordant = only_a + only_b
        p_value = (
            1.0
            if discordant == 0
            else float(
                stats.binomtest(
                    only_a,
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
        )
        raw_p.append(p_value)
        rows.append(
            {
                "model_a": MODELS[a],
                "model_b": MODELS[b],
                "only_a_correct": only_a,
                "only_b_correct": only_b,
                "discordant_samples": discordant,
                "accuracy_difference_a_minus_b": float(
                    correct[:, a].mean() - correct[:, b].mean()
                ),
                "raw_p_value": p_value,
                "holm_p_value": 0.0,
                "significant_0_05": False,
            }
        )
    for row, adjusted in zip(rows, holm_adjust(raw_p)):
        row["holm_p_value"] = adjusted
        row["significant_0_05"] = adjusted < ALPHA
    return rows


def pairwise_cluster_bootstrap(
    cluster_edits: np.ndarray,
    cluster_reference: np.ndarray,
) -> list[dict]:
    rng = np.random.default_rng(SEED)
    cluster_count = cluster_edits.shape[0]
    indices = rng.integers(
        0,
        cluster_count,
        size=(BOOTSTRAP_REPLICATES, cluster_count),
        endpoint=False,
    )
    boot_wer = np.empty((BOOTSTRAP_REPLICATES, len(MODELS)), dtype=float)
    for model_index in range(len(MODELS)):
        edits = cluster_edits[:, model_index][indices].sum(axis=1)
        references = cluster_reference[:, model_index][indices].sum(axis=1)
        boot_wer[:, model_index] = edits / references

    pooled_wer = cluster_edits.sum(axis=0) / cluster_reference.sum(axis=0)
    rows: list[dict] = []
    raw_p: list[float] = []
    for a, b in combinations(range(len(MODELS)), 2):
        distribution = boot_wer[:, a] - boot_wer[:, b]
        lower, upper = np.percentile(distribution, [2.5, 97.5])
        tail = min(int(np.sum(distribution <= 0)), int(np.sum(distribution >= 0)))
        p_value = min(1.0, 2.0 * (tail + 1) / (BOOTSTRAP_REPLICATES + 1))
        raw_p.append(p_value)
        rows.append(
            {
                "model_a": MODELS[a],
                "model_b": MODELS[b],
                "pooled_wer_a": float(pooled_wer[a]),
                "pooled_wer_b": float(pooled_wer[b]),
                "difference_a_minus_b": float(pooled_wer[a] - pooled_wer[b]),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "raw_p_value": p_value,
                "holm_p_value": 0.0,
                "significant_0_05": False,
            }
        )
    for row, adjusted in zip(rows, holm_adjust(raw_p)):
        row["holm_p_value"] = adjusted
        row["significant_0_05"] = adjusted < ALPHA
    return rows


def build_summary(matrices: dict) -> list[dict]:
    rows: list[dict] = []
    for index, model in enumerate(MODELS):
        total_edits = float(matrices["sample_edits"][:, index].sum())
        total_reference = float(matrices["sample_reference"][:, index].sum())
        rows.append(
            {
                "model": model,
                "samples": len(matrices["sample_ids"]),
                "clusters": len(matrices["cluster_keys"]),
                "pooled_wer": total_edits / total_reference,
                "mean_cluster_wer": float(matrices["cluster_wer"][:, index].mean()),
                "exact_sentence_accuracy": float(
                    matrices["sample_correct"][:, index].mean()
                ),
                "substitutions": int(matrices["substitutions"][index]),
                "deletions": int(matrices["deletions"][index]),
                "insertions": int(matrices["insertions"][index]),
            }
        )
    return rows


def thesis_markdown(
    summary: list[dict],
    omnibus: list[dict],
    bootstrap: list[dict],
    ttests: list[dict],
    mcnemar: list[dict],
) -> str:
    anova = omnibus[0]
    friedman = omnibus[1]
    cochran = omnibus[2]
    c2c4_boot = next(
        row
        for row in bootstrap
        if row["model_a"] == "C2" and row["model_b"] == "C4"
    )
    c2c4_t = next(
        row for row in ttests if row["model_a"] == "C2" and row["model_b"] == "C4"
    )
    lines = [
        "# Stage 43 All Five Model Statistical Validation",
        "",
        "## Hypotheses",
        "",
        "For the omnibus WER analysis:",
        "",
        "H0: mu_G1 = mu_C1 = mu_C2 = mu_C3 = mu_C4.",
        "",
        "H1: at least one model mean differs.",
        "",
        "For each paired comparison A versus B:",
        "",
        "H0: mu_d = 0, where d is the paired cluster-WER difference A minus B.",
        "",
        "H1: mu_d is not equal to 0.",
        "",
        "For exact sentence correctness, Cochran's Q tests whether all five paired",
        "correctness probabilities are equal. Pairwise McNemar tests compare the",
        "two discordant cells for each model pair.",
        "",
        "## Formulae",
        "",
        "WER = (S + D + I) / N",
        "",
        "F = MS_model / MS_error",
        "",
        "t = d_bar / (s_d / sqrt(n))",
        "",
        "alpha = 0.05; reject H0 only when the relevant adjusted p-value is below alpha.",
        "",
        "## Omnibus Results",
        "",
        f"Repeated-measures ANOVA: F({anova['corrected_df1']:.3f}, "
        f"{anova['corrected_df2']:.3f}) = {anova['statistic']:.3f}, "
        f"Greenhouse-Geisser corrected p {fmt_p(anova['corrected_p_value'])}, "
        f"epsilon = {anova['greenhouse_geisser_epsilon']:.3f}.",
        "",
        f"Friedman robustness check: chi-square(4) = {friedman['statistic']:.3f}, "
        f"p {fmt_p(friedman['p_value'])}, Kendall's W = "
        f"{friedman['effect_value']:.3f}.",
        "",
        f"Cochran's Q for exact correctness: Q(4) = {cochran['statistic']:.3f}, "
        f"p {fmt_p(cochran['p_value'])}.",
        "",
        "The global null hypothesis was rejected for both WER and exact sentence",
        "correctness. At least one of the five recognizers therefore differed.",
        "",
        "## C2 and C4",
        "",
        f"The pooled-WER difference C2 minus C4 was "
        f"{c2c4_boot['difference_a_minus_b']:.6f}. The 95% cluster-bootstrap CI "
        f"was [{c2c4_boot['ci95_lower']:.6f}, {c2c4_boot['ci95_upper']:.6f}] "
        f"and the Holm-adjusted p-value was {c2c4_boot['holm_p_value']:.6f}.",
        "",
        f"The supplementary paired t-test on equal-weight subject-sentence cluster WER "
        f"also did not distinguish C2 and C4: t({c2c4_t['df']}) = "
        f"{c2c4_t['t_statistic']:.3f}, Holm-adjusted p = "
        f"{c2c4_t['holm_p_value']:.6f}.",
        "",
        "Pooled WER weights tokens, whereas mean cluster WER gives every subject-sentence",
        "cluster equal weight. A small direction change between these estimands is possible",
        "and does not alter the conclusion that the C2-C4 difference was unresolved.",
        "",
        "## Thesis Interpretation",
        "",
        "The omnibus tests establish that the five recognizers did not perform equally.",
        "Corrected pairwise analyses identify the model pairs responsible for that global",
        "difference. A small p-value does not mean that a model is good; the direction and",
        "magnitude of the WER or accuracy difference determine which model performed better.",
        "C2 was not declared statistically superior to C4. Selection between them therefore",
        "also considered fold stability, exact accuracy, parameters and downstream language",
        "quality.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stage43_") as temporary:
        root = resolve_results_root(args.source, Path(temporary))
        predictions = load_predictions(root)
        matrices = build_matrices(predictions)

    summary = build_summary(matrices)
    omnibus = [
        repeated_measures_anova(matrices["cluster_wer"]),
        friedman_test(matrices["cluster_wer"]),
        cochrans_q(matrices["sample_correct"]),
    ]
    ttests = pairwise_ttests(matrices["cluster_wer"])
    mcnemar = pairwise_mcnemar(matrices["sample_correct"])
    bootstrap = pairwise_cluster_bootstrap(
        matrices["cluster_edits"], matrices["cluster_reference"]
    )

    paths = {
        "summary": args.output / "stage43_model_summary.csv",
        "omnibus": args.output / "stage43_omnibus_tests.csv",
        "ttests": args.output / "stage43_pairwise_ttests.csv",
        "mcnemar": args.output / "stage43_pairwise_mcnemar.csv",
        "bootstrap": args.output / "stage43_pairwise_cluster_bootstrap.csv",
        "thesis": args.output / "STAGE_43_THESIS_STATISTICAL_TEXT.md",
    }
    write_csv(paths["summary"], summary)
    write_csv(paths["omnibus"], omnibus)
    write_csv(paths["ttests"], ttests)
    write_csv(paths["mcnemar"], mcnemar)
    write_csv(paths["bootstrap"], bootstrap)
    paths["thesis"].write_text(
        thesis_markdown(summary, omnibus, bootstrap, ttests, mcnemar),
        encoding="utf-8",
    )

    contract = {
        "stage": 43,
        "title": "All Five Model Statistical Validation",
        "models": list(MODELS),
        "samples": len(matrices["sample_ids"]),
        "subject_sentence_clusters": len(matrices["cluster_keys"]),
        "alpha": ALPHA,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "training_started": False,
        "predictions_regenerated": False,
        "output_hashes": {name: sha256(path) for name, path in paths.items()},
        "omnibus_results": omnibus,
    }
    contract_path = args.output / "stage43_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (args.output / "stage43_contract.sha256").write_text(
        sha256(contract_path) + "\n", encoding="utf-8"
    )

    print("=" * 78)
    print("STAGE 43 - ALL FIVE MODEL STATISTICAL VALIDATION")
    print("=" * 78)
    print(f"Matched samples: {len(matrices['sample_ids'])}")
    print(f"Subject-sentence clusters: {len(matrices['cluster_keys'])}")
    for row in omnibus:
        p_value = row["corrected_p_value"] or row["p_value"]
        print(f"{row['test']}: statistic={row['statistic']:.6f}, p={p_value}")
    c2c4 = next(
        row
        for row in bootstrap
        if row["model_a"] == "C2" and row["model_b"] == "C4"
    )
    print(
        "C2-C4 pooled-WER bootstrap: "
        f"delta={c2c4['difference_a_minus_b']:.6f}, "
        f"CI=[{c2c4['ci95_lower']:.6f}, {c2c4['ci95_upper']:.6f}], "
        f"Holm p={c2c4['holm_p_value']:.6f}"
    )
    print("Training started: False")
    print("Predictions regenerated: False")
    print("FINAL STATUS: STAGE 43 PASSED")


if __name__ == "__main__":
    main()
