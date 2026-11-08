# Dataset workspace

Place downloaded datasets under `raw/` and generated/normalized metadata under
`processed/`. Raw dataset contents are intentionally excluded from Git.

For the first integration, use ImgFlip575K's:
- `popular_100_memes.csv`
- `statistics.json`
- `templates/` metadata and clean template images

Do not download the 575,948 individual captioned meme images unless a later
experiment specifically needs them. The application should use the clean
template assets and selected caption examples, not the entire raw corpus at
runtime.
