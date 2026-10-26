from __future__ import annotations

import argparse
import csv
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render claim/image audit contact sheets")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--per-sheet", type=int, default=20)
    args = parser.parse_args()

    review = {
        row["candidate_id"]: row
        for row in _rows(Path(args.review_csv))
        if row.get("decision") == "keep"
    }
    images = _rows(Path(args.image_manifest))
    output_root = Path(args.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    cell_width, cell_height = 560, 390
    columns = 2
    for start in range(0, len(images), args.per_sheet):
        batch = images[start : start + args.per_sheet]
        rows = (len(batch) + columns - 1) // columns
        sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, image_row in enumerate(batch):
            x = (offset % columns) * cell_width
            y = (offset // columns) * cell_height
            claim = review[image_row["candidate_id"]]
            source = Image.open(image_row["local_path"]).convert("RGB")
            fitted = ImageOps.contain(source, (cell_width - 20, 260))
            image_x = x + (cell_width - fitted.width) // 2
            sheet.paste(fitted, (image_x, y + 8))
            header = (
                f"{start + offset + 1:03d} | {image_row['candidate_id']} | "
                f"{claim['normalized_label']} | {claim['decision_cutoff'][:10]}"
            )
            draw.text((x + 8, y + 275), header, fill="black", font=font)
            wrapped = textwrap.wrap(claim["claim_text"], width=66)[:4]
            draw.multiline_text((x + 8, y + 302), "\n".join(wrapped), fill="black", font=small, spacing=2)
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#666666")
        index = start // args.per_sheet + 1
        path = output_root / f"pilot-contact-{index:02d}.jpg"
        sheet.save(path, quality=88)
        print(path)


if __name__ == "__main__":
    main()
