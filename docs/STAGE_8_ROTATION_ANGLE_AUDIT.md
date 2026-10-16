# Stage 8 — Rotation / Angle Audit

## Evidence state

Gu et al. describe the retained values as calculated joint rotation
coordinates.

The current source material does not explicitly establish whether the
export values are degrees or radians.

Therefore numerical scale is treated as empirical evidence, not as
standalone proof of units.

## Dataset

Recordings audited: 7,400

Feature width: 38

Total timesteps: 1,829,320

Total scalar feature values: 69,514,160

## Raw numerical range

Global minimum: -179.997086

Global maximum: 179.99971

Global maximum absolute value: 179.99971

Empirical scale assessment:

`STRONGLY_DEGREE_LIKE`

## Frame-to-frame behaviour

Maximum absolute one-frame change:

359.946503

Jumps greater than 180 raw units:

2,019

Jumps greater than 300 raw units:

1,714

Candidate +/-180 wrap crossings:

1,599

Candidate wrap rate:

0.000023095936

## Important limitation

Candidate wrap detection uses a degree-like diagnostic definition:
opposite signs beyond +/-150 together with a jump of at least 300 raw
units.

This diagnostic must not itself be cited as proof that the export unit
is degrees.

## Column semantics

Exact semantic identities of physical CSV columns 0-37 remain
unverified.

No anatomical label is inferred from numerical behaviour.

## Preprocessing

The primary raw-versus-sin/cos decision is intentionally deferred until
the Stage 8 audit is reviewed.

No raw data were modified.

## Reviewed preprocessing decision

Stage 8 review selected the following policy:

- Primary representation: original 38 raw rotation coordinates.
- Primary temporal unwrapping: no.
- Primary sin/cos encoding: no.
- Raw data modification: no.
- Feature normalization: deferred to Stage 18 and calculated from the
  training partition only.
- Circular representation: declared ablation.
- Circular ablation representation:
  `[sin(theta*pi/180), cos(theta*pi/180)]` for all 38 coordinates,
  producing 76 features.
- No subset of columns will be selected for circular encoding based on
  the full-dataset Stage 8 wrap statistics.

The observed approximately +/-180 range and approximately 360-unit
boundary jumps provide strong evidence of a degree-scaled Euler-angle
representation. The supplied CSV metadata itself does not explicitly
state the unit, so this distinction must be preserved in reporting.

Exact semantic identities of physical CSV columns remain unverified.
