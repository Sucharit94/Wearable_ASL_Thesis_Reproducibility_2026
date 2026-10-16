# Stage 3 — Immutable Dataset Source Setup

## External master

The external master dataset is retained outside the repository in a private,
read-only directory chosen by the researcher.

This folder is retained separately from the fresh project and is not used
directly by project preprocessing, model training, splitting or evaluation.

## Fresh project raw copy

All fresh-project code will read raw source data from:

data/raw/gu_author_dataset

The verified raw structure is:

data/raw/gu_author_dataset/
    label_no_order.txt
    label_order.txt
    csv/
        subject1/
        subject2/
        subject3/

## Verified source-copy facts

- Total source files: 7,402
- CSV recordings: 7,400
- Text label files: 2
- Total bytes: 700,952,400

The project raw copy matches the external master in:

1. relative path structure,
2. filename structure,
3. file sizes,
4. total file count,
5. CSV count,
6. total bytes,
7. both label-file SHA-256 hashes,
8. representative CSV SHA-256.

Representative CSV:

csv/subject1/1_1.csv

SHA-256:

44A73EECF55776B455C852F8BF08D6877C0C9D2A90DC0D27C5A2CC981209E4F7

Recognition label:

label_no_order.txt

SHA-256:

C656D585E5C524964241E62B4B92332C3E744BDC46243FFBBB6F8DF7DCBF3A85

English label:

label_order.txt

SHA-256:

6DD89215C32C2FE238268B0295EFE4B47E5A663C76F9DCC8C1D0B23D431863B0

## Raw-data policy

Files inside data/raw/gu_author_dataset are input-only.

No project process is permitted to overwrite, rename, filter, normalize,
augment, resample or otherwise modify these files in place.

All derived representations must be written elsewhere.

The external master path must not be hard-coded into training or audit code.

## Next stage

Stage 4 will independently enumerate and SHA-256 every CSV recording and
assign stable sample identities.
