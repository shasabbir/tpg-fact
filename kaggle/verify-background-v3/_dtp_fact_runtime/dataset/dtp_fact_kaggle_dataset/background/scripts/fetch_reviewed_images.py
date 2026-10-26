from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from data.image_assets import ImageAssetError, download_image_asset
from data.pilot_review import read_review_csv


MANIFEST_FIELDS = (
    "candidate_id",
    "status",
    "source_url",
    "final_url",
    "local_path",
    "content_type",
    "image_sha256",
    "image_phash",
    "width",
    "height",
    "format",
    "error",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download human-verified target images and calculate exact/perceptual hashes"
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-bytes", type=int, default=25 * 1024 * 1024)
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    for row in read_review_csv(args.review_csv):
        if row.get("decision", "").strip().casefold() != "keep":
            continue
        candidate_id = row.get("candidate_id", "").strip()
        url = row.get("claim_image_url", "").strip()
        try:
            result = download_image_asset(
                url,
                candidate_id,
                args.output_directory,
                max_bytes=args.max_bytes,
            )
            result["error"] = ""
        except ImageAssetError as exc:
            result = {
                "candidate_id": candidate_id,
                "status": "error",
                "source_url": url,
                "error": str(exc),
            }
        results.append(result)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"processed": len(results), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
