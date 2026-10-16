from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence


MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")
SYSTEM_ORDER = ("L0", "L1", "L2")


_TEXT_HASH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


def sha256_file(path: str | Path) -> str:
    """Hash text canonically so Git line-ending conversion cannot break evidence."""
    target = Path(path)
    digest = hashlib.sha256()
    if target.suffix.lower() in _TEXT_HASH_SUFFIXES:
        data = target.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
        return digest.hexdigest()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(payload))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def model_id(row: Mapping[str, object]) -> str:
    for key, value in row.items():
        if norm(key) in {"model", "model_id", "recognizer", "recognizer_id"}:
            model = norm(value)
            if model in MODEL_ORDER:
                return model
    raise RuntimeError(f"Could not identify model in row with fields {list(row)}")


def first_value(row: Mapping[str, object], aliases: Sequence[str]) -> str:
    index = {norm(key): str(value).strip() for key, value in row.items()}
    for alias in aliases:
        value = index.get(norm(alias), "")
        if value != "":
            return value
    return ""


def as_float(value: object, *, default: float | None = None) -> float:
    text = str(value).strip().replace("%", "")
    if text == "":
        if default is None:
            raise RuntimeError("Required numeric value is missing")
        return default
    result = float(text)
    if "%" in str(value):
        result /= 100.0
    return result


def index_models(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for row in rows:
        key = model_id(row)
        if key in output:
            raise RuntimeError(f"Duplicate model row: {key}")
        output[key] = row
    return output


def normalize_language_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    expected = {(model, system) for model in MODEL_ORDER for system in SYSTEM_ORDER}
    output: list[dict[str, object]] = []
    observed: set[tuple[str, str]] = set()
    for row in rows:
        model = model_id(row)
        system = first_value(row, ("system_id", "system")).upper()
        if system not in SYSTEM_ORDER:
            raise RuntimeError(f"Unexpected language system: {system!r}")
        key = (model, system)
        if key in observed:
            raise RuntimeError(f"Duplicate language summary row: {key}")
        observed.add(key)
        output.append(
            {
                "model_id": model,
                "recognizer_role": "quality_reference" if model == "g1" else ("primary_lightweight" if model == "c2" else "comparator"),
                "system_id": system,
                "system_name": {"L0": "raw normalized gloss", "L1": "training-only retrieval", "L2": "constrained FLAN-T5 correction"}[system],
                "samples": int(as_float(first_value(row, ("samples",)))),
                "pooled_wer": as_float(first_value(row, ("pooled_wer", "wer"))),
                "wer_ci_lower": as_float(first_value(row, ("pooled_wer_ci_lower", "wer_ci_lower")), default=float("nan")),
                "wer_ci_upper": as_float(first_value(row, ("pooled_wer_ci_upper", "wer_ci_upper")), default=float("nan")),
                "exact_sentence_accuracy": as_float(first_value(row, ("exact_sentence_accuracy", "exact_accuracy", "accuracy"))),
                "content_precision": as_float(first_value(row, ("content_precision",))),
                "content_recall": as_float(first_value(row, ("content_recall",))),
                "content_f1": as_float(first_value(row, ("content_f1", "content_f_score"))),
                "unsupported_content_rate": as_float(first_value(row, ("unsupported_content_rate",))),
                "correction_harm_rate": first_value(row, ("correction_harm_rate",)),
            }
        )
    if observed != expected:
        raise RuntimeError(f"Stage 40 summary coverage mismatch; missing={sorted(expected-observed)}")
    output.sort(key=lambda row: (MODEL_ORDER.index(str(row["model_id"])), SYSTEM_ORDER.index(str(row["system_id"]))))
    for system in SYSTEM_ORDER:
        ranked = sorted((row for row in output if row["system_id"] == system), key=lambda row: (float(row["pooled_wer"]), -float(row["content_f1"])))
        for rank, row in enumerate(ranked, 1):
            row["quality_rank_within_system"] = rank
    return output


def _metric(row: Mapping[str, object], aliases: Sequence[str]) -> str:
    return first_value(row, aliases)


def build_recognition_table(
    publication_rows: Sequence[Mapping[str, object]],
    efficiency_rows: Sequence[Mapping[str, object]],
    stability_rows: Sequence[Mapping[str, object]],
    error_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    publication = index_models(publication_rows)
    efficiency = index_models(efficiency_rows)
    stability = index_models(stability_rows)
    errors = index_models(error_rows)
    for name, values in (("publication", publication), ("efficiency", efficiency), ("stability", stability), ("error", errors)):
        if set(values) != set(MODEL_ORDER):
            raise RuntimeError(f"Stage 41 requires five {name} rows; found {sorted(values)}")
    param_aliases = ("parameters", "parameter_count", "trainable_parameters", "params", "number_of_parameters")
    g1_parameters = as_float(_metric(efficiency["g1"], param_aliases))
    output: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        pub, eff, stable, err = publication[model], efficiency[model], stability[model], errors[model]
        parameters = as_float(_metric(eff, param_aliases))
        reduction = 100.0 * (1.0 - parameters / g1_parameters)
        output.append(
            {
                "model_id": model,
                "role": "recognition_quality_reference" if model == "g1" else ("recommended_lightweight_recognizer" if model == "c2" else "comparator"),
                "mean_pooled_wer": _metric(stable, ("mean_wer", "pooled_wer_mean", "wer_mean", "pooled_wer")) or _metric(pub, ("pooled_wer", "wer")),
                "wer_standard_deviation": _metric(stable, ("std_wer", "wer_std", "pooled_wer_std", "wer_standard_deviation")),
                "wer_range": _metric(stable, ("wer_range", "range_wer")),
                "exact_sentence_accuracy": _metric(pub, ("exact_sentence_accuracy", "accuracy", "mean_accuracy")),
                "ser": _metric(pub, ("ser", "sentence_error_rate")),
                "parameter_count": int(parameters),
                "parameter_reduction_vs_g1_percent": reduction,
                "model_size_mb": _metric(eff, ("model_size_mb", "serialized_size_mb", "size_mb")),
                "latency_ms": _metric(eff, ("latency_ms", "median_latency_ms", "inference_latency_ms")),
                "throughput_samples_per_second": _metric(eff, ("throughput_samples_per_second", "throughput_sps", "samples_per_second")),
                "peak_memory_mb": _metric(eff, ("peak_memory_mb", "memory_mb", "peak_gpu_memory_mb")),
                "dominant_error_type": _metric(err, ("dominant_error", "dominant_error_type")),
            }
        )
    return output


def comparison_winner(row: Mapping[str, object], left_key: str, right_key: str) -> str:
    conclusion = first_value(row, ("wer_conclusion", "conclusion"))
    if conclusion == "a_lower_wer":
        return str(row[left_key]).lower()
    if conclusion == "b_lower_wer":
        return str(row[right_key]).lower()
    return "not_resolved"


def find_system_test(rows: Sequence[Mapping[str, object]], model: str, a: str, b: str) -> Mapping[str, object]:
    for row in rows:
        if model_id(row) != model:
            continue
        left = first_value(row, ("system_a",)).upper()
        right = first_value(row, ("system_b",)).upper()
        if {left, right} == {a, b}:
            return row
    raise RuntimeError(f"Missing paired Stage 40 comparison for {model}: {a} vs {b}")


def find_model_test(rows: Sequence[Mapping[str, object]], a: str, b: str) -> Mapping[str, object]:
    for row in rows:
        left = first_value(row, ("model_a",)).lower()
        right = first_value(row, ("model_b",)).lower()
        if {left, right} == {a, b}:
            return row
    raise RuntimeError(f"Missing paired Stage 40 L2 comparison for {a} vs {b}")


def build_final_selection(language_rows: Sequence[Mapping[str, object]], system_tests: Sequence[Mapping[str, object]], l2_tests: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    lookup = {(str(row["model_id"]), str(row["system_id"])): row for row in language_rows}
    c2_test = find_system_test(system_tests, "c2", "L1", "L2")
    l2_c2_c4 = find_model_test(l2_tests, "c2", "c4")
    return [
        {
            "decision": "recognition_quality_reference",
            "selected_model": "G1",
            "selected_language_system": "n.a.",
            "primary_evidence": "lowest five-fold recognition WER and highest exact accuracy",
            "statistical_context": "G1 was significantly better in the Stage 34 paired comparisons",
            "claim_scope": "known-signer mixed-subject five-fold recognition",
        },
        {
            "decision": "lightweight_recognizer",
            "selected_model": "C2",
            "selected_language_system": "n.a.",
            "primary_evidence": "best joint recognition-quality, size, latency, throughput, memory and stability trade-off",
            "statistical_context": "C2-C4 recognition WER difference was not resolved at 95%; efficiency breaks the tie",
            "claim_scope": "recommended lightweight model, not the absolute accuracy winner",
        },
        {
            "decision": "closed_template_language_method",
            "selected_model": "all recognizers",
            "selected_language_system": "L1",
            "primary_evidence": "lowest English WER for every recognizer in the evaluated closed-template protocol",
            "statistical_context": f"C2 L1-vs-L2 WER conclusion: {comparison_winner(c2_test, 'system_a', 'system_b')}",
            "claim_scope": "training-only retrieval on the known 300-sentence inventory",
        },
        {
            "decision": "lightweight_end_to_end_system",
            "selected_model": "C2",
            "selected_language_system": "L1",
            "primary_evidence": f"WER={float(lookup[('c2','L1')]['pooled_wer']):.6f}, exact={float(lookup[('c2','L1')]['exact_sentence_accuracy']):.6f}, content F1={float(lookup[('c2','L1')]['content_f1']):.4f}",
            "statistical_context": "selection combines language quality with the frozen lightweight efficiency evidence",
            "claim_scope": "final recommended lightweight closed-template pipeline",
        },
        {
            "decision": "generative_language_reference",
            "selected_model": "G1",
            "selected_language_system": "L2",
            "primary_evidence": f"best L2 WER={float(lookup[('g1','L2')]['pooled_wer']):.6f}",
            "statistical_context": f"C2-vs-C4 L2 WER conclusion: {comparison_winner(l2_c2_c4, 'model_a', 'model_b')}",
            "claim_scope": "best constrained-generation result; not the lightweight recommendation",
        },
    ]


def build_claim_register(language_rows: Sequence[Mapping[str, object]], final_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    lookup = {(str(row["model_id"]), str(row["system_id"])): row for row in language_rows}
    c2_l1, c2_l2 = lookup[("c2", "L1")], lookup[("c2", "L2")]
    return [
        {"claim_id": "S41-01", "status": "supported", "thesis_safe_claim": "G1 is the recognition-quality reference under known-signer mixed-subject five-fold evaluation.", "evidence": "Stages 34-36", "prohibited_overclaim": "G1 is a lightweight model."},
        {"claim_id": "S41-02", "status": "supported", "thesis_safe_claim": "C2 is the recommended lightweight recognizer after considering quality, parameters, size, latency, throughput, memory and fold stability.", "evidence": "Stages 34D-36", "prohibited_overclaim": "C2 is the overall accuracy winner."},
        {"claim_id": "S41-03", "status": "supported", "thesis_safe_claim": f"C2+L1 achieved English WER {float(c2_l1['pooled_wer']):.6f}, exact sentence accuracy {float(c2_l1['exact_sentence_accuracy']):.4%}, and content F1 {float(c2_l1['content_f1']):.4f}.", "evidence": "Stage 40", "prohibited_overclaim": "This proves open-vocabulary translation."},
        {"claim_id": "S41-04", "status": "supported", "thesis_safe_claim": f"C2+L2 achieved English WER {float(c2_l2['pooled_wer']):.6f}, exact sentence accuracy {float(c2_l2['exact_sentence_accuracy']):.4%}, and content F1 {float(c2_l2['content_f1']):.4f}.", "evidence": "Stage 40", "prohibited_overclaim": "The generator outperformed retrieval."},
        {"claim_id": "S41-05", "status": "supported", "thesis_safe_claim": "English WER was the primary language metric; exact accuracy, content precision, recall and F1, unsupported-content rate, harm rate, confidence intervals and paired tests were also reported.", "evidence": "Stage 40", "prohibited_overclaim": "Accuracy alone determined the winner."},
        {"claim_id": "S41-06", "status": "limitation", "thesis_safe_claim": "The very high L1 result reflects a closed 300-sentence inventory with fold-specific training-only retrieval banks.", "evidence": "Stages 37-40", "prohibited_overclaim": "L1 generalizes to unseen sentence templates."},
        {"claim_id": "S41-07", "status": "not_executed", "thesis_safe_claim": "L3 template-holdout evaluation remains future work.", "evidence": "Stage 37 protocol", "prohibited_overclaim": "Unseen-template performance was evaluated."},
        {"claim_id": "S41-08", "status": "limitation", "thesis_safe_claim": "Unsupported-content rate measures lexical support relative to the source gloss; it is not a complete human hallucination assessment.", "evidence": "Stage 40 metric definition", "prohibited_overclaim": "Unsupported-content rate alone proves semantic faithfulness."},
    ]


def markdown_report(recognition: Sequence[Mapping[str, object]], language: Sequence[Mapping[str, object]], selections: Sequence[Mapping[str, object]], claims: Sequence[Mapping[str, object]]) -> str:
    by = {(str(row["model_id"]), str(row["system_id"])): row for row in language}
    lines = [
        "# Stage 41: Final thesis evidence package", "",
        "Stage 41 consolidates the frozen known-signer recognition, efficiency, statistical and language evidence. It does not train a recognizer, regenerate predictions or call a language model.", "",
        "## Final model and system decisions", "",
        "| Decision | Selection | Interpretation |", "|---|---|---|",
    ]
    for row in selections:
        selection = str(row["selected_model"]) + (" + " + str(row["selected_language_system"]) if row["selected_language_system"] != "n.a." else "")
        lines.append(f"| {row['decision']} | {selection} | {row['claim_scope']} |")
    lines += ["", "## Recognition results", "", "| Model | Role | Mean WER | WER SD | Exact accuracy | Parameters | Reduction vs G1 | Dominant error |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in recognition:
        exact = str(row["exact_sentence_accuracy"] or "n.a.")
        lines.append(f"| {str(row['model_id']).upper()} | {row['role']} | {row['mean_pooled_wer']} | {row['wer_standard_deviation']} | {exact} | {int(row['parameter_count']):,} | {float(row['parameter_reduction_vs_g1_percent']):.2f}% | {row['dominant_error_type']} |")
    lines += ["", "## End-to-end English results", "", "| Model | L0 WER / exact / F1 | L1 WER / exact / F1 | L2 WER / exact / F1 |", "|---|---|---|---|"]
    for model in MODEL_ORDER:
        cells = []
        for system in SYSTEM_ORDER:
            row = by[(model, system)]
            cells.append(f"{float(row['pooled_wer']):.6f} / {float(row['exact_sentence_accuracy']):.4%} / {float(row['content_f1']):.4f}")
        lines.append(f"| {model.upper()} | " + " | ".join(cells) + " |")
    lines += [
        "", "## Thesis-ready interpretation", "",
        "G1 produced the best recognition quality and therefore serves as the quality reference. C2 is the recommended lightweight recognizer because the C2-C4 recognition WER difference was not resolved at 95%, while the frozen efficiency and stability evidence favored C2. The choice therefore does not rely on accuracy alone.", "",
        f"For the final lightweight pipeline, C2 with L1 retrieval achieved WER {float(by[('c2','L1')]['pooled_wer']):.6f}, exact sentence accuracy {float(by[('c2','L1')]['exact_sentence_accuracy']):.4%}, content precision {float(by[('c2','L1')]['content_precision']):.4f}, content recall {float(by[('c2','L1')]['content_recall']):.4f}, and content F1 {float(by[('c2','L1')]['content_f1']):.4f}. L1 outperformed the constrained generator in this closed-template experiment.", "",
        "The L1 result must be described as closed-template retrieval performance. The corpus contains a fixed 300-sentence inventory, and each retrieval bank was built only from the training portion of its fold. The evaluation prevented outer-test reference leakage, but it did not test unseen sentence templates or open-vocabulary translation.", "",
        "## Limitations and future work", "",
        "- L3 template-holdout evaluation was specified but not executed.",
        "- Results apply to known-signer mixed-subject five-fold evaluation.",
        "- Exact accuracy is strict and can penalize valid paraphrases; content F1 supplies a complementary semantic-content view.",
        "- Unsupported-content rate is a lexical support diagnostic, not a substitute for human semantic evaluation.",
        "- The study compares three participants and a closed sentence inventory, so broader signer and vocabulary generalization remains future work.", "",
        "## Claim register", "", "| ID | Status | Safe claim | Prohibited overclaim |", "|---|---|---|---|",
    ]
    for row in claims:
        lines.append(f"| {row['claim_id']} | {row['status']} | {row['thesis_safe_claim']} | {row['prohibited_overclaim']} |")
    return "\n".join(lines) + "\n"


def relative_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
