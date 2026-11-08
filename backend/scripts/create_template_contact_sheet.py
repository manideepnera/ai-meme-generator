#!/usr/bin/env python3
"""Create a numbered contact sheet for staged ImgFlip template images.

Run from the backend directory:
    python scripts/create_template_contact_sheet.py

The script only reads the staging manifest/images and writes a review image.
It does not enable templates or modify the active template catalogue.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BACKEND_DIR / "datasets" / "processed" / "imgflip_templates" / "manifest.json"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=210)
    parser.add_argument("--thumb-height", type=int, default=165)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.columns < 1 or args.thumb_width < 50 or args.thumb_height < 50:
        parser.error("columns and thumbnail dimensions must be positive (at least 50px)")

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    templates = data.get("templates", [])
    if not templates:
        raise SystemExit("The manifest contains no templates.")

    image_root = manifest_path.parent
    label_height = 48
    padding = 12
    cell_width = args.thumb_width + padding * 2
    cell_height = args.thumb_height + label_height + padding * 2
    rows = math.ceil(len(templates) / args.columns)
    sheet = Image.new(
        "RGB",
        (args.columns * cell_width, rows * cell_height),
        "#eeeeee",
    )
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(15)
    detail_font = load_font(12)

    missing: list[str] = []
    for index, item in enumerate(templates, start=1):
        relative_image = item.get("image")
        image_path = image_root / relative_image if relative_image else None
        if image_path is None or not image_path.is_file():
            missing.append(f"{item.get('name', 'unknown')} ({relative_image})")
            continue

        try:
            with Image.open(image_path) as source:
                source.load()
                thumbnail = ImageOps.contain(
                    source.convert("RGB"),
                    (args.thumb_width, args.thumb_height),
                )
        except Exception as exc:
            missing.append(f"{item.get('name', 'unknown')} ({exc})")
            continue

        slot = index - 1
        col = slot % args.columns
        row = slot // args.columns
        x = col * cell_width
        y = row * cell_height

        draw.rounded_rectangle(
            (x + 3, y + 3, x + cell_width - 3, y + cell_height - 3),
            radius=8,
            fill="white",
            outline="#cccccc",
            width=1,
        )
        image_x = x + padding + (args.thumb_width - thumbnail.width) // 2
        image_y = y + padding + (args.thumb_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (image_x, image_y))

        name = str(item.get("name", "Unnamed template"))
        if len(name) > 29:
            name = name[:26] + "..."
        label_y = y + padding + args.thumb_height + 4
        draw.text((x + padding, label_y), f"{index:03d}. {name}", fill="#111111", font=title_font)
        draw.text(
            (x + padding, label_y + 21),
            f"ID: {item.get('template_id', 'unknown')}",
            fill="#555555",
            font=detail_font,
        )

    output_path = args.output or (manifest_path.parent / "contact_sheet.jpg")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90, optimize=True)

    print(f"Templates in manifest: {len(templates)}")
    print(f"Contact sheet: {output_path}")
    print(f"Grid: {args.columns} columns x {rows} rows")
    print(f"Missing/unreadable images: {len(missing)}")
    for entry in missing:
        print(f"  - {entry}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
