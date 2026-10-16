# Stage 7 — Feature-Semantics Audit

## Result

The raw recordings contain 38 numeric columns.

Gu et al. (2024), Table 2, establishes the semantic inventory of the
38 selected joint-rotation coordinates.

However, the current author-provided dataset contains no CSV header or
separate feature-order metadata that establishes which exact physical
CSV column corresponds to each semantic coordinate.

Therefore:

- semantic inventory: VERIFIED
- raw physical column order: VERIFIED
- exact semantic identity of CSV columns 0-37: UNVERIFIED
- anatomical 2D column topology: UNVERIFIED

## Safe terminology

The input is described as:

`wearable inertial motion-capture-derived joint-rotation features`

or:

`T x 38 joint-rotation feature sequence`

The project must not describe these CSV inputs as sEMG.

The current files also do not establish that the 38 retained values
are raw accelerometer, gyroscope, or magnetometer channels.

## Swin implication

The primary 2D representation may use the native source feature order
as a temporal-feature map.

It must not claim that vertical feature adjacency represents verified
anatomical adjacency unless authoritative column-order metadata is
obtained later.

## Evidence rule

Table 2 defines membership in the 38-dimensional semantic feature set.

Table 2 printed order is not treated as proof of physical CSV column
order.

No semantic mapping will be inferred merely from numeric magnitude or
correlation.
