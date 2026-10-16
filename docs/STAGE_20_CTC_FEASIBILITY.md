# Stage 20 — CTC Feasibility

## Minimum CTC length

For target sequence `y`, the minimum CTC temporal length is:

`target length + number of adjacent repeated target pairs`

This accounts for the blank required to separate identical adjacent
target labels.

## Dataset

Recordings audited:

7,400

Sentence templates:

300

Maximum target-token length:

8

Maximum adjacent repeated-pair count:

0

Maximum minimum CTC requirement:

8

## Direct source sequences

CTC-infeasible raw source sequences:

0

This is only source-level feasibility.

Later model-specific downsampling must still propagate valid output
lengths exactly.

## Windowed G1/C4 input

Window length:

36

Reference stride:

18

Reference-stride infeasible recordings:

0

## Frozen selected stride

Selected stride:

18

Selected-stride infeasible recordings:

0

Selection rule:

Choose the largest integer stride not greater than 18 that provides
zero CTC-infeasible recordings over all 7,400 samples.

## Scientific policy

No recording may be silently removed merely because a temporal
windowing configuration is CTC-infeasible.

The Stage-20 selected window rule is therefore frozen before model
training.

## Output-length boundary

Stage 20 freezes input/window-level feasibility.

Stage 22 will propagate exact valid lengths through each implemented
model architecture.
