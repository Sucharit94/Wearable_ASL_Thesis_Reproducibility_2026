# Stage 41: Final thesis evidence package

Stage 41 consolidates the frozen known-signer recognition, efficiency, statistical and language evidence. It does not train a recognizer, regenerate predictions or call a language model.

## Final model and system decisions

| Decision | Selection | Interpretation |
|---|---|---|
| recognition_quality_reference | G1 | known-signer mixed-subject five-fold recognition |
| lightweight_recognizer | C2 | recommended lightweight model, not the absolute accuracy winner |
| closed_template_language_method | all recognizers + L1 | training-only retrieval on the known 300-sentence inventory |
| lightweight_end_to_end_system | C2 + L1 | final recommended lightweight closed-template pipeline |
| generative_language_reference | G1 + L2 | best constrained-generation result; not the lightweight recommendation |

## Recognition results

| Model | Role | Mean WER | WER SD | Exact accuracy | Parameters | Reduction vs G1 | Dominant error |
|---|---|---:|---:|---:|---:|---:|---|
| G1 | recognition_quality_reference | 0.001073729420186113 |  | 0.9964864864864865 | 4,791,688 | 0.00% | deletion |
| C1 | comparator | 0.06721546170365068 |  | 0.7995945945945946 | 392,488 | 91.81% | deletion |
| C2 | recommended_lightweight_recognizer | 0.04005010737294202 |  | 0.8647297297297297 | 656,168 | 86.31% | deletion |
| C3 | comparator | 0.44466714387974227 |  | 0.13648648648648654 | 237,352 | 95.05% | deletion |
| C4 | comparator | 0.04090909090909091 |  | 0.8766216216216216 | 348,604 | 92.72% | deletion |

## End-to-end English results

| Model | L0 WER / exact / F1 | L1 WER / exact / F1 | L2 WER / exact / F1 |
|---|---|---|---|
| G1 | 0.546866 / 9.7027% / 0.7966 | 0.000058 / 99.9865% / 1.0000 | 0.085451 / 81.9730% / 0.9616 |
| C1 | 0.569206 / 8.6757% / 0.7716 | 0.018373 / 97.5270% / 0.9878 | 0.149597 / 70.6622% / 0.9234 |
| C2 | 0.562047 / 8.2432% / 0.7815 | 0.004715 / 98.7973% / 0.9958 | 0.117855 / 76.1622% / 0.9444 |
| C3 | 0.701380 / 3.4054% / 0.5713 | 0.300431 / 64.0811% / 0.7801 | 0.578608 / 18.6216% / 0.6253 |
| C4 | 0.559546 / 9.2973% / 0.7830 | 0.008453 / 98.8784% / 0.9942 | 0.127631 / 75.6216% / 0.9377 |

## Thesis-ready interpretation

G1 produced the best recognition quality and therefore serves as the quality reference. C2 is the recommended lightweight recognizer because the C2-C4 recognition WER difference was not resolved at 95%, while the frozen efficiency and stability evidence favored C2. The choice therefore does not rely on accuracy alone.

For the final lightweight pipeline, C2 with L1 retrieval achieved WER 0.004715, exact sentence accuracy 98.7973%, content precision 0.9966, content recall 0.9950, and content F1 0.9958. L1 outperformed the constrained generator in this closed-template experiment.

The L1 result must be described as closed-template retrieval performance. The corpus contains a fixed 300-sentence inventory, and each retrieval bank was built only from the training portion of its fold. The evaluation prevented outer-test reference leakage, but it did not test unseen sentence templates or open-vocabulary translation.

## Limitations and future work

- L3 template-holdout evaluation was specified but not executed.
- Results apply to known-signer mixed-subject five-fold evaluation.
- Exact accuracy is strict and can penalize valid paraphrases; content F1 supplies a complementary semantic-content view.
- Unsupported-content rate is a lexical support diagnostic, not a substitute for human semantic evaluation.
- The study compares three participants and a closed sentence inventory, so broader signer and vocabulary generalization remains future work.

## Claim register

| ID | Status | Safe claim | Prohibited overclaim |
|---|---|---|---|
| S41-01 | supported | G1 is the recognition-quality reference under known-signer mixed-subject five-fold evaluation. | G1 is a lightweight model. |
| S41-02 | supported | C2 is the recommended lightweight recognizer after considering quality, parameters, size, latency, throughput, memory and fold stability. | C2 is the overall accuracy winner. |
| S41-03 | supported | C2+L1 achieved English WER 0.004715, exact sentence accuracy 98.7973%, and content F1 0.9958. | This proves open-vocabulary translation. |
| S41-04 | supported | C2+L2 achieved English WER 0.117855, exact sentence accuracy 76.1622%, and content F1 0.9444. | The generator outperformed retrieval. |
| S41-05 | supported | English WER was the primary language metric; exact accuracy, content precision, recall and F1, unsupported-content rate, harm rate, confidence intervals and paired tests were also reported. | Accuracy alone determined the winner. |
| S41-06 | limitation | The very high L1 result reflects a closed 300-sentence inventory with fold-specific training-only retrieval banks. | L1 generalizes to unseen sentence templates. |
| S41-07 | not_executed | L3 template-holdout evaluation remains future work. | Unseen-template performance was evaluated. |
| S41-08 | limitation | Unsupported-content rate measures lexical support relative to the source gloss; it is not a complete human hallucination assessment. | Unsupported-content rate alone proves semantic faithfulness. |
