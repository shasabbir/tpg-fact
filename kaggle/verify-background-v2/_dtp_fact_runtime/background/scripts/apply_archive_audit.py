from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply independent archive outcomes to evidence candidates")
    parser.add_argument("--evidence-review", required=True)
    parser.add_argument("--archive-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_csv(Path(args.evidence_review))
    archive = {row["evidence_id"]: row for row in read_csv(Path(args.archive_audit))}
    counts = {"keep": 0, "reject": 0}
    for row in rows:
        audit = archive.get(row["evidence_id"], {})
        if audit.get("status") == "observed_pre_cutoff":
            row["decision"] = "keep"
            row["exclusion_reason"] = ""
            row["archived_at"] = audit.get("archived_at", "")
            row["timestamp_confidence"] = "observed"
            row["reviewer_notes"] = (
                row.get("reviewer_notes", "")
                + " Independent pre-cutoff observation confirmed by "
                + audit.get("archive_source", "open_archive")
                + "."
            ).strip()
            counts["keep"] += 1
        else:
            row["decision"] = "reject"
            row["exclusion_reason"] = audit.get("status", "missing_archive_audit")
            row["archived_at"] = ""
            row["timestamp_confidence"] = "uncertain"
            counts["reject"] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"kept={counts['keep']}")
    print(f"rejected={counts['reject']}")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
