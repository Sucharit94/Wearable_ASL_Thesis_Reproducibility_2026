# Dataset access and placement

The Gu, Oku and Todoh dataset is not redistributed in this repository. Obtain
it from the original authors or its authorized distribution point and follow
the accompanying terms, participant-data restrictions, and citation guidance.

The dataset link is provided in the thesis appendix because public redistribution is restricted.

Expected local layout:

```text
data/raw/gu_author_dataset/
├── label_no_order.txt
├── label_order.txt
└── csv/
    ├── subject1/
    ├── subject2/
    └── subject3/
```

The verified study copy contained 7,400 CSV recordings and two label files
(700,952,400 bytes total). Before running preprocessing, confirm the label
hashes recorded in `docs/STAGE_3_SOURCE_PROVENANCE.md` and run the source and
manifest audit scripts. Raw files are input-only: do not rename, overwrite,
normalize, augment, or filter them in place.

The repository's `.gitignore` excludes `data/`; do not force-add participant
recordings to GitHub. Dataset users remain responsible for the original data
licence and ethics requirements.
