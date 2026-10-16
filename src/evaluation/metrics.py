from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral


def _validate_tokens(
    values: Sequence[int],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:

    tokens = tuple(values)

    if not allow_empty and not tokens:
        raise ValueError(
            f"{name} cannot be empty."
        )

    validated = []

    for value in tokens:

        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
        ):
            raise TypeError(
                f"{name} must contain integer token IDs."
            )

        token = int(value)

        if token <= 0:
            raise ValueError(
                f"{name} contains CTC blank, padding, "
                "or a negative token ID."
            )

        validated.append(token)

    return tuple(validated)


def levenshtein_alignment(
    reference: Sequence[int],
    hypothesis: Sequence[int],
) -> list[dict[str, int | str | None]]:
    """
    Deterministic token-level Levenshtein alignment.

    Tie-breaking precedence during backtracking:

    1. exact match;
    2. substitution;
    3. deletion;
    4. insertion.

    Freezing this order makes S/D/I decomposition reproducible
    when several minimum-distance alignments exist.
    """

    ref = _validate_tokens(
        reference,
        name="reference",
        allow_empty=False,
    )

    hyp = _validate_tokens(
        hypothesis,
        name="hypothesis",
        allow_empty=True,
    )

    n = len(ref)
    m = len(hyp)

    distance = [
        [
            0
            for _ in range(m + 1)
        ]
        for _ in range(n + 1)
    ]

    for i in range(n + 1):
        distance[i][0] = i

    for j in range(m + 1):
        distance[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if ref[i - 1] == hyp[j - 1]:

                distance[i][j] = (
                    distance[i - 1][j - 1]
                )

            else:

                distance[i][j] = 1 + min(
                    distance[i - 1][j - 1],
                    distance[i - 1][j],
                    distance[i][j - 1],
                )

    operations = []

    i = n
    j = m

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and ref[i - 1] == hyp[j - 1]
            and distance[i][j]
            == distance[i - 1][j - 1]
        ):

            operations.append(
                {
                    "operation": "match",
                    "reference_index": i - 1,
                    "hypothesis_index": j - 1,
                    "reference_token": ref[i - 1],
                    "hypothesis_token": hyp[j - 1],
                }
            )

            i -= 1
            j -= 1
            continue

        if (
            i > 0
            and j > 0
            and distance[i][j]
            == distance[i - 1][j - 1] + 1
        ):

            operations.append(
                {
                    "operation": "substitution",
                    "reference_index": i - 1,
                    "hypothesis_index": j - 1,
                    "reference_token": ref[i - 1],
                    "hypothesis_token": hyp[j - 1],
                }
            )

            i -= 1
            j -= 1
            continue

        if (
            i > 0
            and distance[i][j]
            == distance[i - 1][j] + 1
        ):

            operations.append(
                {
                    "operation": "deletion",
                    "reference_index": i - 1,
                    "hypothesis_index": None,
                    "reference_token": ref[i - 1],
                    "hypothesis_token": None,
                }
            )

            i -= 1
            continue

        if (
            j > 0
            and distance[i][j]
            == distance[i][j - 1] + 1
        ):

            operations.append(
                {
                    "operation": "insertion",
                    "reference_index": None,
                    "hypothesis_index": j - 1,
                    "reference_token": None,
                    "hypothesis_token": hyp[j - 1],
                }
            )

            j -= 1
            continue

        raise RuntimeError(
            "Levenshtein backtracking reached "
            "an inconsistent state."
        )

    operations.reverse()

    return operations


def score_sequence(
    reference: Sequence[int],
    hypothesis: Sequence[int],
) -> dict:

    ref = _validate_tokens(
        reference,
        name="reference",
        allow_empty=False,
    )

    hyp = _validate_tokens(
        hypothesis,
        name="hypothesis",
        allow_empty=True,
    )

    alignment = levenshtein_alignment(
        ref,
        hyp,
    )

    substitutions = sum(
        operation["operation"] == "substitution"
        for operation in alignment
    )

    deletions = sum(
        operation["operation"] == "deletion"
        for operation in alignment
    )

    insertions = sum(
        operation["operation"] == "insertion"
        for operation in alignment
    )

    matches = sum(
        operation["operation"] == "match"
        for operation in alignment
    )

    edit_distance = (
        substitutions
        + deletions
        + insertions
    )

    wer = (
        edit_distance
        / len(ref)
    )

    exact_match = (
        ref == hyp
    )

    return {
        "reference_tokens": list(ref),
        "hypothesis_tokens": list(hyp),
        "reference_length": len(ref),
        "hypothesis_length": len(hyp),
        "matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "edit_distance": edit_distance,
        "wer": wer,
        "exact_match": exact_match,
        "sentence_error": int(
            not exact_match
        ),
        "alignment": alignment,
    }


def score_batch(
    references: Sequence[Sequence[int]],
    hypotheses: Sequence[Sequence[int]],
) -> dict:
    """
    Calculate per-sample results and pooled recognition metrics.

    Headline WER is pooled:

        sum(S + D + I) / sum(reference tokens)

    It is not the unweighted average of sample WER values.
    """

    reference_list = list(references)
    hypothesis_list = list(hypotheses)

    if not reference_list:
        raise ValueError(
            "Metric batch cannot be empty."
        )

    if len(reference_list) != len(
        hypothesis_list
    ):
        raise ValueError(
            "Reference and hypothesis batch "
            "sizes do not match."
        )

    sample_results = []

    for sample_index, (
        reference,
        hypothesis,
    ) in enumerate(
        zip(
            reference_list,
            hypothesis_list,
            strict=True,
        )
    ):

        result = score_sequence(
            reference,
            hypothesis,
        )

        result["sample_index"] = (
            sample_index
        )

        sample_results.append(
            result
        )

    reference_tokens = sum(
        result["reference_length"]
        for result in sample_results
    )

    hypothesis_tokens = sum(
        result["hypothesis_length"]
        for result in sample_results
    )

    substitutions = sum(
        result["substitutions"]
        for result in sample_results
    )

    deletions = sum(
        result["deletions"]
        for result in sample_results
    )

    insertions = sum(
        result["insertions"]
        for result in sample_results
    )

    matches = sum(
        result["matches"]
        for result in sample_results
    )

    total_edits = (
        substitutions
        + deletions
        + insertions
    )

    sentence_errors = sum(
        result["sentence_error"]
        for result in sample_results
    )

    sample_count = len(
        sample_results
    )

    pooled_wer = (
        total_edits
        / reference_tokens
    )

    ser = (
        sentence_errors
        / sample_count
    )

    macro_sample_wer = sum(
        result["wer"]
        for result in sample_results
    ) / sample_count

    aggregate = {
        "samples": sample_count,
        "reference_tokens": reference_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        "matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_edits": total_edits,
        "pooled_wer": pooled_wer,
        "one_minus_wer": 1.0 - pooled_wer,
        "macro_sample_wer": macro_sample_wer,
        "sentence_errors": sentence_errors,
        "ser": ser,
        "exact_sentence_accuracy": 1.0 - ser,
    }

    return {
        "aggregate": aggregate,
        "samples": sample_results,
    }
