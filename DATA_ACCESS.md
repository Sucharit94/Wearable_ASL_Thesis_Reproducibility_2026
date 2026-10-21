# Dataset access and placement

The Gu, Oku and Todoh dataset is not redistributed in this repository. Obtain
it from the original authors or its authorized distribution point and follow
the accompanying terms, participant-data restrictions, and citation guidance.

## Access for thesis examination

The researcher plans to provide authorized examiners with controlled access to
the study copy through Google Drive. The access details will be supplied in
the examination copy of the thesis and are intentionally omitted from this
public repository. No dataset download link is published here.

Availability is subject to the original dataset owner's permission, applicable
data-use terms, and institutional requirements. Until an authorized examination
copy is available, obtain the dataset from the original authors or an
authorized distribution point. Possession of the thesis or its link does not
itself grant permission to redistribute the participant recordings.

The researcher should restrict the Drive folder to authorized examination
accounts. Keeping a link out of GitHub does not prevent a thesis recipient
from forwarding it; access permissions should enforce the intended audience.
Code, frozen predictions, generation responses, and the documented analysis
verification route remain publicly available without raw-data access.

## Dataset placement and integrity

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
