# GitHub publication notes

This candidate was created from committed Git revision a75336202dcfaa3b2db0c7233da15cb2b1f5de8a.

## Included in the candidate

- Committed source code, tests, configurations, documentation, figures and small analysis artifacts.
- The Stage 39B V2 Colab notebook and companion cell when they were found.
- The recorded Colab environment files when they were found.

## Kept outside the repository

- Raw dataset files.
- Python virtual environments and caches.
- 198 PyTorch checkpoint files from the downloaded Colab folder.
- The 265 MiB Stage 17 binary cache.
- Duplicate transfer bundles and copied run directories.
- Frozen prediction archives and Stage 39/43 result packages intended for a GitHub Release or research archive.

The separate release-assets directory contains selected compact evidence files and a SHA-256 manifest. Do not copy the complete Colab folder into this repository.

## Required review before publication

1. Replace the planning-only README with final installation and reproduction instructions.
2. Decide whether to publish the dataset access instructions rather than the raw dataset.
3. Add a licence and citation file after confirming the intended reuse terms.
4. Correct the Stage 40/41 cross-platform line-ending hash issue before claiming Linux reproducibility.
5. Inspect Untitled0.ipynb in the release-assets manual-review folder before deciding whether it belongs in the repository.
