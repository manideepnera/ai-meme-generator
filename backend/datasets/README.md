# Dataset workspace

Keep downloaded source files under `raw/` and normalized application-ready data under `processed/`. Raw files are excluded from Git.

## ImgFlip575K source layout

Source repository: https://github.com/schesa/ImgFlip575K_Dataset

The source describes 575,948 meme examples and provides:
- `dataset/popular_100_memes.csv`: top-100 popular template list
- `dataset/statistics.json`: number of examples per template
- `dataset/templates/`: one JSON metadata file per template, including its title, ImgFlip template URL, alternative names, template ID and image dimensions
- `dataset/memes/`: individual captioned meme examples

Important: the template JSON files contain metadata and a `template_url`; do not assume they include the actual image bytes. The `memes/` directory contains the much larger example corpus and is not needed for the first integration.

## Recommended local layout

```text
backend/datasets/
  raw/
    imgflip575k/
      dataset/
        popular_100_memes.csv
        statistics.json
        templates/
          *.json
  processed/
    imgflip_templates/
      manifest.json
      images/
        <template-id>.<jpg|png>
```

Download only the top-100 CSV, statistics file and template metadata first. Then fetch/cache the clean template images for the subset we decide to support. Keep the original source files unchanged in `raw/`.

## Integration policy

Do not import all 575,948 captioned meme examples into the runtime or commit them to this application repository. Begin by reviewing the 100 template candidates, then curate a smaller reliable set. Each enabled template must have:
- a downloaded and verified clean image
- stable source template ID, title and alternative names
- exact text-slot count and semantic slot names
- manually checked slot coordinates and character limits
- a semantic profile describing purpose, narrative pattern, suitable intents and cases to avoid

The existing renderer needs local image files and explicit slot coordinates. ImgFlip's source metadata alone is not enough to enable a template automatically.

After download, the next implementation step is an importer that converts the source CSV/JSON into a normalized manifest, downloads only approved template images, and maps curated templates into the application's existing template configuration and semantic catalog.
