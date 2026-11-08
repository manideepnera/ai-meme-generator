"""Stage the ImgFlip575K popular-100 templates for manual review.

This script does NOT register templates in the live application. Every imported
entry remains disabled until its text slots and coordinates are reviewed.
Run from the backend directory:
    python scripts/import_imgflip_templates.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BACKEND_DIR / "datasets"
SOURCE_DIR = DATASET_DIR / "raw" / "imgflip575k" / "dataset"
OUTPUT_DIR = DATASET_DIR / "processed" / "imgflip_templates"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"


def normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_source() -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], dict[str, int]]:
    csv_path = SOURCE_DIR / "popular_100_memes.csv"
    stats_path = SOURCE_DIR / "statistics.json"
    templates_dir = SOURCE_DIR / "templates"

    for required in (csv_path, stats_path, templates_dir):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Follow backend/datasets/README.md to prepare the source dataset."
            )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        popular = list(csv.DictReader(handle))

    stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = stats_data.get("memes", {})

    metadata_by_id: dict[str, dict[str, Any]] = {}
    metadata_by_name: dict[str, dict[str, Any]] = {}
    for metadata_path in templates_dir.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping invalid metadata {metadata_path.name}: {exc}")
            continue
        template_id = str(metadata.get("template_id", "")).strip()
        if template_id:
            metadata_by_id[template_id] = metadata
        metadata_by_name[normalise_name(metadata_path.stem)] = metadata

    return popular, metadata_by_id | {}, stats


def main() -> None:
    popular, metadata_by_id, stats = load_source()
    templates_dir = SOURCE_DIR / "templates"
    source_images = templates_dir / "img"
    if not source_images.is_dir():
        raise FileNotFoundError(f"Template image directory not found: {source_images}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a metadata filename index to support entries with missing IDs.
    metadata_by_name: dict[str, tuple[Path, dict[str, Any]]] = {}
    for metadata_path in templates_dir.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata_by_name[normalise_name(metadata_path.stem)] = (metadata_path, metadata)

    entries: list[dict[str, Any]] = []
    missing: list[str] = []

    for row in popular:
        template_id = str(row.get("ID", "")).strip()
        display_name = str(row.get("Name", "")).strip()
        aliases_csv = str(row.get("Alternate Names", row.get("Alternate Names", ""))).strip()

        metadata = metadata_by_id.get(template_id)
        metadata_path: Path | None = None
        if metadata:
            for candidate in templates_dir.glob("*.json"):
                try:
                    candidate_data = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if str(candidate_data.get("template_id", "")).strip() == template_id:
                    metadata_path = candidate
                    break

        if not metadata:
            matched = metadata_by_name.get(normalise_name(display_name))
            if matched:
                metadata_path, metadata = matched

        if not metadata or not metadata_path:
            missing.append(f"{template_id} — {display_name} (metadata missing)")
            continue

        image_name = metadata_path.stem
        image_path = next(
            (p for p in source_images.glob(f"{image_name}.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            None,
        )
        if image_path is None:
            missing.append(f"{template_id} — {display_name} (clean image missing)")
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = (image.format or image_path.suffix.lstrip(".")).lower()
        except Exception as exc:
            missing.append(f"{template_id} — {display_name} (invalid image: {exc})")
            continue

        extension = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
        destination_name = f"{template_id}{extension}"
        destination = IMAGE_OUTPUT_DIR / destination_name
        shutil.copy2(image_path, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()

        source_aliases = metadata.get("alternative_names", "")
        aliases = [item.strip() for item in source_aliases.split(",") if item.strip()]
        if aliases_csv:
            aliases.extend(item.strip() for item in aliases_csv.split(",") if item.strip())
        aliases = list(dict.fromkeys(aliases))

        entries.append({
            "template_id": template_id,
            "name": display_name or metadata.get("title", metadata_path.stem),
            "metadata_title": metadata.get("title", ""),
            "alternate_names": aliases,
            "source_metadata_file": metadata_path.name,
            "source_image_file": image_path.name,
            "source_image_url": metadata.get("template_url", ""),
            "image": f"images/{destination_name}",
            "width": width,
            "height": height,
            "format": image_format,
            "source_example_count": stats.get(metadata_path.name, 0),
            "sha256": digest,
            "integration_status": "needs_slot_review",
            "enabled": False,
            "text_slots": [],
            "semantic_profile": None,
        })

    manifest = {
        "source": "https://github.com/schesa/ImgFlip575K_Dataset",
        "source_total_examples": 575948,
        "popular_template_count": len(popular),
        "imported_template_count": len(entries),
        "missing_count": len(missing),
        "policy": "Staging manifest only. Templates are disabled until slot geometry and semantic profiles are reviewed.",
        "templates": entries,
        "missing": missing,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Imported {len(entries)} of {len(popular)} popular templates.")
    print(f"Manifest: {manifest_path}")
    print(f"Images:   {IMAGE_OUTPUT_DIR}")
    if missing:
        print("\nEntries needing attention:")
        for item in missing:
            print(f" - {item}")
    print("\nNo live template configuration was changed.")


if __name__ == "__main__":
    main()
