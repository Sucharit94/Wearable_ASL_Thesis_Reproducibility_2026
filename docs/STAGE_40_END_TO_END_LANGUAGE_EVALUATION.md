# Stage 40: End-to-end language evaluation

Stage 40 evaluates the same 7,400 paired outer-fold samples for each of the five recognizers. L0 is raw normalized gloss, L1 is deterministic training-only retrieval, and L2 is the frozen FLAN-T5 correction output. No recognizer training or language generation is performed here.

## All-model results

| Model | System | English WER | Exact accuracy | Content precision | Content recall | Content F1 | Unsupported content | Harm vs L0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | L0 | 0.546866 | 9.7027% | 0.7694 | 0.8259 | 0.7966 | 0.0000% | n.a. |
| G1 | L1 | 0.000058 | 99.9865% | 1.0000 | 1.0000 | 1.0000 | 17.3397% | 0.0000% |
| G1 | L2 | 0.085451 | 81.9730% | 0.9654 | 0.9577 | 0.9616 | 15.4219% | 3.2432% |
| C1 | L0 | 0.569206 | 8.6757% | 0.7655 | 0.7779 | 0.7716 | 0.0000% | n.a. |
| C1 | L1 | 0.018373 | 97.5270% | 0.9923 | 0.9832 | 0.9878 | 21.5875% | 0.5946% |
| C1 | L2 | 0.149597 | 70.6622% | 0.9470 | 0.9009 | 0.9234 | 15.5378% | 3.2027% |
| C2 | L0 | 0.562047 | 8.2432% | 0.7593 | 0.8050 | 0.7815 | 0.0000% | n.a. |
| C2 | L1 | 0.004715 | 98.7973% | 0.9966 | 0.9950 | 0.9958 | 19.3196% | 0.0811% |
| C2 | L2 | 0.117855 | 76.1622% | 0.9526 | 0.9364 | 0.9444 | 15.8090% | 3.6486% |
| C3 | L0 | 0.701380 | 3.4054% | 0.7441 | 0.4636 | 0.5713 | 0.0000% | n.a. |
| C3 | L1 | 0.300431 | 64.0811% | 0.8488 | 0.7218 | 0.7801 | 47.2166% | 10.3243% |
| C3 | L2 | 0.578608 | 18.6216% | 0.8156 | 0.5070 | 0.6253 | 14.2681% | 3.7432% |
| C4 | L0 | 0.559546 | 9.2973% | 0.7670 | 0.7997 | 0.7830 | 0.0000% | n.a. |
| C4 | L1 | 0.008453 | 98.8784% | 0.9961 | 0.9923 | 0.9942 | 19.7212% | 0.3919% |
| C4 | L2 | 0.127631 | 75.6216% | 0.9526 | 0.9234 | 0.9377 | 15.3083% | 3.5405% |

## L2 end-to-end quality ranking

The rank uses pooled English WER as the primary criterion, content F1 as the first tie-breaker, and exact sentence accuracy as the second. It is a language-quality rank; the Stage 34 efficiency evidence remains necessary for a lightweight deployment decision.

| Rank | Model | WER | Exact accuracy | Content F1 | Unsupported content | Harm vs L0 | Frozen role |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | G1 | 0.085451 | 81.9730% | 0.9616 | 15.4219% | 3.2432% | recognition quality reference |
| 2 | C2 | 0.117855 | 76.1622% | 0.9444 | 15.8090% | 3.6486% | primary lightweight recognizer |
| 3 | C4 | 0.127631 | 75.6216% | 0.9377 | 15.3083% | 3.5405% | comparator |
| 4 | C1 | 0.149597 | 70.6622% | 0.9234 | 15.5378% | 3.2027% | comparator |
| 5 | C3 | 0.578608 | 18.6216% | 0.6253 | 14.2681% | 3.7432% | comparator |

## Statistical method

- 10,000 paired cluster-bootstrap replicates with subject-sentence clusters.
- Exact two-sided McNemar tests for paired sentence correctness.
- Holm correction applied separately to the 15 within-model system comparisons and the 10 L2 model comparisons.
- WER differences unresolved at 95%: 0 within-model system comparisons and 1 L2 model comparisons.

## Metric definitions

- English WER is total word substitutions, deletions, and insertions divided by reference words.
- Exact accuracy requires equality after frozen lowercase, punctuation-stripping, underscore-splitting normalization.
- Content precision, recall, and F1 use multiset token overlap after removing the frozen function-word list.
- Unsupported-content rate is the proportion of predicted content tokens absent from the source-gloss content-token set.
- Correction-harm rate is the proportion of samples whose system word-edit count is worse than L0 for the same sample.

## Claim boundary

The L2 ranking measures end-to-end English output quality under the known-signer five-fold protocol. It does not replace the recognition-only comparison or the efficiency measurements, and it does not establish template-holdout performance.
