# Dataset workspace

Keep downloaded source files under `raw/` and normalized application-ready data under `processed/`. Both directories are excluded from Git.

## ImgFlip575K source layout

Source repository: https://github.com/schesa/ImgFlip575K_Dataset

The source describes 575,948 meme examples and provides:
- `dataset/popular_100_memes.csv`: top-100 popular template list
- `dataset/statistics.json`: number of examples per template
- `dataset/templates/*.json`: template metadata, including title, ImgFlip URL, alternative names, template ID and dimensions
- `dataset/templates/img/`: clean template JPG images
- `dataset/memes/`: individual captioned meme examples

The metadata JSON and clean template image are separate files. The image filename generally matches the metadata JSON filename.

## Local layout

After sparse checkout:

```text
backend/datasets/
  raw/
    imgflip575k/
      dataset/
        popular_100_memes.csv
        statistics.json
        templates/
          *.json
          img/
            *.jpg
  processed/
    imgflip_templates/
      manifest.json
      images/
        <template-id>.<jpg|png>
```

Only the top-100 CSV, statistics, template metadata and clean template images are needed for this first integration. Do not check out or import `dataset/memes/`.

## Build the staging manifest

From the `backend/` directory, run:

```powershell
python scripts/import_imgflip_templates.py
```

The importer:
1. Reads the top-100 CSV and statistics.
2. Matches each popular template to its metadata JSON by template ID (with a name fallback).
3. Verifies the corresponding clean image and copies it into `processed/imgflip_templates/images/`.
4. Writes `processed/imgflip_templates/manifest.json` with source IDs, names, aliases, image dimensions, usage counts and SHA-256 checksums.
5. Reports any missing metadata or images.

The output is a **staging manifest only**. It does not modify `meme_templates/index.json` or the live semantic catalog.

## Integration policy

Do not import all 575,948 captioned meme examples into the runtime or commit raw/processed dataset files to this application repository. Review the 100 template candidates, then curate a smaller reliable set. Each enabled template must have:
- a verified clean image
- stable source template ID, title and alternative names
- exact text-slot count and semantic slot names
- manually checked slot coordinates and character limits
- a semantic profile describing purpose, narrative pattern, suitable intents and cases to avoid

The current renderer requires local image files and explicit slot coordinates. Dataset metadata does not provide verified text-slot geometry. Imported templates therefore remain disabled until their slot layout and semantic profile are reviewed.
