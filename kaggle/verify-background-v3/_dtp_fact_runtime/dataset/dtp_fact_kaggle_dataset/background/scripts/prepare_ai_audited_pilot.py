from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import html as html_module
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from data.image_assets import inspect_image_bytes
from data.labels import normalize_rating
from data.pilot_review import REVIEW_FIELDS
from retrieval.evidence_candidates import default_evidence_query


USER_AGENT = (
    "Mozilla/5.0 (compatible; DTP-Fact/0.1; "
    "+mailto:26-94030-1@student.aiub.edu)"
)
VISUAL_RE = re.compile(
    r"\b(photo(?:graph)?|image|picture|video|screenshot|graphic|map|chart|meme|"
    r"clip|footage|selfie|poster|sign|tweet|post|reel)\b",
    re.IGNORECASE,
)
US_RE = re.compile(
    r"\b(U\.?S\.?|United States|America(?:n)?|President|Congress|Senate|"
    r"White House|Biden|Trump|Obama|Harris|Clinton|Bush|Florida|Texas|California|"
    r"New York|Georgia|Washington|Arizona|Pennsylvania|Michigan|Ohio|Kentucky|"
    r"Tennessee|Missouri|FBI|CIA|CDC|Supreme Court|Republican|Democrat|"
    r"Capitol|Pentagon|US Army|US Navy)\b",
    re.IGNORECASE,
)
BAD_IMAGE_RE = re.compile(
    r"(logo|rating-|icon|avatar|author|placeholder|pixel|badge)", re.IGNORECASE
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _request(url: str, timeout: int = 35) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    with urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get("Content-Type", "")


def _jsonld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for node in soup.find_all("script", type="application/ld+json"):
        text = node.string or node.get_text("", strip=True)
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                objects.extend(child for child in graph if isinstance(child, dict))
            objects.append(item)
    return objects


def _type_matches(item: dict[str, Any], expected: str) -> bool:
    value = item.get("@type")
    values = value if isinstance(value, list) else [value]
    return expected in values


def _image_from_article(article: dict[str, Any], soup: BeautifulSoup) -> str:
    candidates: list[str] = []
    for key in ("thumbnailUrl", "image"):
        value = article.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict) and isinstance(item.get("url"), str):
                candidates.append(item["url"])
    for key in ("og:image", "twitter:image"):
        node = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if node and node.get("content"):
            candidates.append(str(node["content"]))
    for candidate in candidates:
        candidate = html_module.unescape(candidate.strip())
        if candidate.startswith("http") and not BAD_IMAGE_RE.search(candidate):
            return candidate
    return ""


def _page_record(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_id": row["candidate_id"],
        "status": "error",
        "review_url": row["review_url"],
    }
    try:
        payload, _ = _request(row["review_url"])
        soup = BeautifulSoup(payload, "html.parser")
        objects = _jsonld_objects(soup)
        article = next((item for item in objects if _type_matches(item, "Article")), {})
        claim_review = next((item for item in objects if _type_matches(item, "ClaimReview")), {})
        claim_reviewed = html_module.unescape(str(claim_review.get("claimReviewed") or "")).strip()
        rating = claim_review.get("reviewRating") or {}
        structured_rating = str(rating.get("alternateName") or "").strip() if isinstance(rating, dict) else ""
        published_at = str(article.get("datePublished") or claim_review.get("datePublished") or "").strip()
        image_url = _image_from_article(article, soup)
        headline = html_module.unescape(str(article.get("headline") or "")).strip()
        description = html_module.unescape(str(article.get("description") or "")).strip()
        result.update(
            {
                "status": "ok",
                "claim_reviewed": claim_reviewed,
                "structured_rating": structured_rating,
                "published_at": published_at,
                "image_url": image_url,
                "headline": headline,
                "description": description,
                "us_signal": bool(US_RE.search(" ".join((claim_reviewed, headline, description)))),
            }
        )
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
        result["error"] = str(error)
    return result


def _normalized_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", value)
        if len(token) > 2
    }


def _similarity(left: str, right: str) -> float:
    a = _normalized_tokens(left)
    b = _normalized_tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _event_id(claim: str, published_at: str) -> str:
    tokens = sorted(_normalized_tokens(claim))[:12]
    year = published_at[:4]
    return "evt-" + hashlib.sha256((year + "|" + "|".join(tokens)).encode("utf-8")).hexdigest()[:12]


def _download_image(row: dict[str, Any], image_root: Path) -> dict[str, Any]:
    result = dict(row)
    try:
        data, content_type = _request(str(row["image_url"]), timeout=45)
        details = inspect_image_bytes(data)
        if details["width"] < 300 or details["height"] < 180:
            raise ValueError(f"image_too_small:{details['width']}x{details['height']}")
        extension = ".png" if "png" in content_type.casefold() else ".jpg"
        path = image_root / f"{row['candidate_id']}{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result.update(
            {
                "image_status": "ok",
                "local_path": str(path.resolve()),
                **details,
            }
        )
    except Exception as error:
        result.update({"image_status": "error", "image_error": str(error)})
    return result


def _round_robin(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        published = datetime.fromisoformat(str(row["published_at"]).replace("Z", "+00:00"))
        year = published.year
        split = "train" if year <= 2023 else "validation" if year == 2024 else "test"
        buckets.setdefault((str(row["normalized_label"]), split), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (str(item["published_at"]), str(item["candidate_id"])))
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < target and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < target:
                selected.append(buckets[key].pop(0))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a high-precision, machine-audited Snopes pilot shortlist"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--target", type=int, default=240)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    source_rows = _read_csv(Path(args.input))
    candidates = [
        row
        for row in source_rows
        if row.get("publisher_site") == "snopes.com"
        and row.get("suggested_label")
        and VISUAL_RE.search(row.get("claim_text", ""))
        and (
            US_RE.search(" ".join((row.get("claim_text", ""), row.get("review_title", ""))))
            or row.get("suggested_label") == "insufficient_evidence"
        )
    ]
    output_root = Path(args.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    page_records: list[dict[str, Any]] = []
    by_id = {row["candidate_id"]: row for row in candidates}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_page_record, row): row["candidate_id"] for row in candidates}
        for position, future in enumerate(as_completed(futures), start=1):
            page_records.append(future.result())
            if position % 50 == 0:
                print(f"fetched {position}/{len(candidates)}", flush=True)

    eligible: list[dict[str, Any]] = []
    for page in page_records:
        source = by_id[page["candidate_id"]]
        suggested = source.get("suggested_label", "")
        structured = normalize_rating(str(page.get("structured_rating", "")))
        claim_match = _similarity(source.get("claim_text", ""), str(page.get("claim_reviewed", "")))
        reasons: list[str] = []
        if page.get("status") != "ok":
            reasons.append("page_fetch_failed")
        if not page.get("published_at"):
            reasons.append("missing_publication_time")
        if not page.get("image_url"):
            reasons.append("missing_article_image")
        if structured != suggested:
            reasons.append("structured_rating_mismatch")
        if claim_match < 0.55:
            reasons.append("claimreview_text_mismatch")
        if not page.get("us_signal"):
            reasons.append("us_scope_not_confirmed")
        page.update(
            {
                "normalized_label": suggested,
                "claim_similarity": claim_match,
                "eligibility_reasons": "|".join(reasons),
            }
        )
        if not reasons:
            eligible.append(page)

    shortlist = _round_robin(eligible, max(args.target + 40, args.target))
    image_root = output_root / "images"
    downloaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(2, args.workers // 2)) as executor:
        futures = [executor.submit(_download_image, row, image_root) for row in shortlist]
        for future in as_completed(futures):
            downloaded.append(future.result())
    usable = [row for row in downloaded if row.get("image_status") == "ok"]
    selected = _round_robin(usable, args.target)
    selected_ids = {str(row["candidate_id"]) for row in selected}
    page_by_id = {str(row["candidate_id"]): row for row in page_records}

    review_rows: list[dict[str, str]] = []
    for row in candidates:
        result = {field: row.get(field, "") for field in REVIEW_FIELDS}
        page = page_by_id.get(row["candidate_id"], {})
        if row["candidate_id"] in selected_ids:
            result.update(
                {
                    "decision": "keep",
                    "exclusion_reason": "",
                    "us_relevant": "yes",
                    "multimodal": "yes",
                    "target_image_verified": "yes",
                    "rating_verified": "yes",
                    "claim_image_url": str(page["image_url"]),
                    "claim_time": str(page["published_at"]),
                    "decision_cutoff": str(page["published_at"]),
                    "cutoff_basis": "earliest_verified_fact_check_publication",
                    "cutoff_confidence": "confirmed",
                    "normalized_label": row["suggested_label"],
                    "event_id": _event_id(row["claim_text"], str(page["published_at"])),
                    "evidence_query": default_evidence_query(row["claim_text"]),
                    "sufficiency_target": "unlabeled",
                    "sufficiency_target_source": "",
                    "reviewer_notes": (
                        "Machine audit: structured ClaimReview text/rating matched; US signal "
                        "matched page metadata; page Article image downloaded and decoded. "
                        "Human visual confirmation remains pending."
                    ),
                }
            )
        else:
            reasons = str(page.get("eligibility_reasons") or "outside_pilot_quota")
            result.update(
                {
                    "decision": "reject",
                    "exclusion_reason": reasons,
                    "reviewer_notes": "Machine screening rejection; no benchmark inclusion.",
                }
            )
        review_rows.append(result)

    image_fields = [
        "candidate_id", "status", "source_url", "final_url", "local_path",
        "content_type", "image_sha256", "image_phash", "width", "height",
        "format", "error",
    ]
    image_manifest_rows = [
        {
            "candidate_id": row["candidate_id"],
            "status": row["image_status"],
            "source_url": row["image_url"],
            "final_url": row["image_url"],
            "local_path": row["local_path"],
            "content_type": "image/" + str(row.get("format", "jpeg")).casefold(),
            "image_sha256": row["image_sha256"],
            "image_phash": row["image_phash"],
            "width": row["width"],
            "height": row["height"],
            "format": row["format"],
            "error": "",
        }
        for row in selected
    ]
    _write_csv(output_root / "pilot_review_machine_audited.csv", list(REVIEW_FIELDS), review_rows)
    _write_csv(
        output_root / "image_manifest_machine_audited.csv",
        image_fields,
        image_manifest_rows,
    )
    _write_csv(
        output_root / "page_audit.csv",
        [
            "candidate_id", "status", "review_url", "claim_reviewed", "structured_rating",
            "published_at", "image_url", "headline", "description", "us_signal",
            "normalized_label", "claim_similarity", "eligibility_reasons", "error",
        ],
        sorted(page_records, key=lambda row: str(row["candidate_id"])),
    )
    manifest = {
        "status": "machine_audited_visual_confirmation_pending",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(Path(args.input).resolve()),
        "candidate_rows_considered": len(candidates),
        "pages_fetched": sum(row.get("status") == "ok" for row in page_records),
        "metadata_eligible": len(eligible),
        "downloaded_images": len(usable),
        "selected_rows": len(selected),
        "label_counts": {
            label: sum(row.get("normalized_label") == label for row in selected)
            for label in ("supported", "refuted", "insufficient_evidence")
        },
        "warning": (
            "The machine audit verifies metadata and decodable article imagery. It does not "
            "constitute human visual confirmation or a final benchmark release."
        ),
    }
    (output_root / "machine_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
