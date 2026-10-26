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
from typing import Any
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from data.image_assets import inspect_image_bytes
from data.labels import normalize_rating
from data.pilot_review import REVIEW_FIELDS
from retrieval.evidence_candidates import default_evidence_query


DEFAULT_UA = "Mozilla/5.0 (compatible; DTP-Fact/0.1; +mailto:26-94030-1@student.aiub.edu)"
AFP_UA = "facebookexternalhit/1.1"
VISUAL_RE = re.compile(
    r"\b(photo(?:graph)?|image|picture|video|screenshot|graphic|map|chart|meme|"
    r"clip|footage|selfie|poster|sign|tweet|post|reel)\b",
    re.IGNORECASE,
)
US_RE = re.compile(
    r"\b(U\.?S\.?|United States|America(?:n)?|Trump|Biden|Obama|Harris|Clinton|"
    r"Bush|Congress|Senate|White House|Social Security|Medicare|Medicaid|FBI|CIA|"
    r"CDC|Pentagon|Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|"
    r"Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|"
    r"Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|"
    r"Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|"
    r"North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|"
    r"South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
    r"West Virginia|Wisconsin|Wyoming|District of Columbia|Supreme Court|Republican|"
    r"Democrat|Capitol|JD Vance|Musk|Federal Reserve|National Park Service)\b",
    re.IGNORECASE,
)
ARCHIVE_RE = re.compile(
    r"https?://web\.archive\.org/web/(\d{8,14})(?:id_)?/(https?://.+)",
    re.IGNORECASE,
)
BAD_IMAGE_RE = re.compile(
    r"(logo|rating-|icon|avatar|author|placeholder|pixel|badge|600x200|newsletter)",
    re.IGNORECASE,
)
QUOTAS = {"supported": 42, "insufficient_evidence": 70, "refuted": 168}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def request(url: str, *, afp: bool = False, timeout: int = 40) -> tuple[bytes, str]:
    req = Request(
        url,
        headers={
            "User-Agent": AFP_UA if afp else DEFAULT_UA,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        return response.read(), response.geturl()


def jsonld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for node in soup.find_all("script", type="application/ld+json"):
        text = node.string or node.get_text("", strip=True)
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        queue = value if isinstance(value, list) else [value]
        for item in queue:
            if not isinstance(item, dict):
                continue
            objects.append(item)
            graph = item.get("@graph")
            if isinstance(graph, list):
                objects.extend(child for child in graph if isinstance(child, dict))
    return objects


def type_matches(item: dict[str, Any], expected: str) -> bool:
    value = item.get("@type")
    values = value if isinstance(value, list) else [value]
    return expected in values


def parse_datetime(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def archive_time(value: str) -> datetime:
    padded = value + "0" * (14 - len(value))
    formats = {8: "%Y%m%d", 14: "%Y%m%d%H%M%S"}
    return datetime.strptime(padded, formats[len(padded)]).replace(tzinfo=timezone.utc)


def tokens(value: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", value) if len(token) > 2}


def similarity(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def event_id(claim: str, published_at: str) -> str:
    stable = "|".join(sorted(tokens(claim))[:12])
    return "evt-" + hashlib.sha256((published_at[:4] + "|" + stable).encode()).hexdigest()[:12]


def image_candidates(
    row: dict[str, str], soup: BeautifulSoup, article: BeautifulSoup, publisher: str
) -> list[str]:
    og = soup.find("meta", property="og:image")
    og_url = urljoin(row["review_url"], str(og.get("content", ""))) if og else ""
    candidates: list[str] = []
    if publisher == "factcheck.afp.com":
        for image in article.find_all("img"):
            url = urljoin(
                row["review_url"],
                str(image.get("src") or image.get("data-src") or ""),
            )
            if not url or BAD_IMAGE_RE.search(url):
                continue
            if og_url and urlparse(url).path.rsplit("/", 1)[-1] == urlparse(og_url).path.rsplit("/", 1)[-1]:
                continue
            candidates.append(url)
        if og_url:
            candidates.append(og_url)
    elif og_url and not BAD_IMAGE_RE.search(og_url):
        candidates.append(og_url)
    return list(dict.fromkeys(candidates))


def page_record(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"candidate_id": row["candidate_id"], "status": "error"}
    publisher = row["publisher_site"]
    try:
        payload, _ = request(row["review_url"], afp=publisher == "factcheck.afp.com")
        soup = BeautifulSoup(payload, "html.parser")
        article = (
            soup.find("article", id="article-content")
            or soup.find("article")
            or soup.find("main")
        )
        if article is None:
            raise ValueError("missing_article_container")
        objects = jsonld_objects(soup)
        review = next((item for item in objects if type_matches(item, "ClaimReview")), {})
        rating = review.get("reviewRating") or {}
        structured_rating = str(rating.get("alternateName") or "") if isinstance(rating, dict) else ""
        claim_reviewed = html_module.unescape(str(review.get("claimReviewed") or row["claim_text"]))
        published_at = str(review.get("datePublished") or row["api_review_date"])
        cutoff = parse_datetime(published_at)
        evidence: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for anchor in article.find_all("a", href=True):
            url = html_module.unescape(str(anchor["href"]).strip())
            match = ARCHIVE_RE.match(url)
            if not match:
                continue
            observed = archive_time(match.group(1))
            if observed > cutoff:
                continue
            original_url = unquote(match.group(2))
            if original_url in seen_urls:
                continue
            seen_urls.add(original_url)
            host = urlparse(original_url).netloc.casefold().removeprefix("www.")
            if not host:
                continue
            parent = anchor.find_parent(["p", "li", "blockquote"])
            context = parent.get_text(" ", strip=True) if parent else anchor.get_text(" ", strip=True)
            evidence.append(
                {
                    "archive_url": url,
                    "original_url": original_url,
                    "domain": host,
                    "archived_at": observed.isoformat().replace("+00:00", "Z"),
                    "context": context,
                }
            )
        article_text = article.get_text(" ", strip=True)
        us_scope_text = " ".join(
            (
                row["claim_text"],
                claim_reviewed,
                str(review.get("name") or ""),
            )
        )
        result.update(
            {
                "status": "ok",
                "claim_reviewed": claim_reviewed,
                "structured_rating": structured_rating,
                "published_at": cutoff.isoformat().replace("+00:00", "Z"),
                "headline": str(review.get("name") or ""),
                "article_text": article_text,
                # Scope must come from the claim metadata itself.  Searching the
                # article body produced false positives whenever a non-US story
                # merely mentioned the United States in background prose.
                "us_signal": bool(US_RE.search(us_scope_text)),
                "claim_similarity": similarity(row["claim_text"], claim_reviewed),
                "image_candidates": image_candidates(row, soup, article, publisher),
                "evidence": evidence,
                "evidence_domains": sorted({item["domain"] for item in evidence}),
            }
        )
    except Exception as error:
        result["error"] = str(error)
    return result


def download_first_image(record: dict[str, Any], root: Path) -> dict[str, Any]:
    result = dict(record)
    errors: list[str] = []
    for url in record["image_candidates"]:
        try:
            data, final_url = request(url, afp=record["publisher_site"] == "factcheck.afp.com")
            details = inspect_image_bytes(data)
            if details["width"] < 300 or details["height"] < 180:
                raise ValueError(f"image_too_small:{details['width']}x{details['height']}")
            suffix = ".png" if str(details["format"]).casefold() == "png" else ".jpg"
            path = root / f"{record['candidate_id']}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            result.update(
                {
                    "image_status": "ok",
                    "claim_image_url": final_url,
                    "local_path": str(path.resolve()),
                    **details,
                }
            )
            return result
        except Exception as error:
            errors.append(f"{url}:{error}")
    result.update({"image_status": "error", "image_error": " | ".join(errors)})
    return result


def split(record: dict[str, Any]) -> str:
    year = parse_datetime(record["published_at"]).year
    return "train" if year <= 2023 else "validation" if year == 2024 else "test"


def select(rows: list[dict[str, Any]], quotas: dict[str, int]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label, quota in quotas.items():
        candidates = [row for row in rows if row["normalized_label"] == label]
        candidates.sort(
            key=lambda row: (
                len(row["evidence_domains"]) >= 2,
                len(row["evidence_domains"]),
                split(row) == "test",
                row["published_at"],
            ),
            reverse=True,
        )
        # Round-robin among temporal partitions without sacrificing the evidence-first order.
        buckets = {name: [row for row in candidates if split(row) == name] for name in ("train", "validation", "test")}
        picked: list[dict[str, Any]] = []
        order = ("test", "train", "validation")
        while len(picked) < quota and any(buckets.values()):
            for name in order:
                if buckets[name] and len(picked) < quota:
                    picked.append(buckets[name].pop(0))
        selected.extend(picked)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an evidence-first multimodal pilot buffer")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--supported-quota", type=int, default=QUOTAS["supported"])
    parser.add_argument(
        "--insufficient-quota", type=int, default=QUOTAS["insufficient_evidence"]
    )
    parser.add_argument("--refuted-quota", type=int, default=QUOTAS["refuted"])
    args = parser.parse_args()

    quotas = {
        "supported": args.supported_quota,
        "insufficient_evidence": args.insufficient_quota,
        "refuted": args.refuted_quota,
    }

    source_rows = read_csv(Path(args.input))
    candidates = [
        row
        for row in source_rows
        if (
            row["publisher_site"] == "factcheck.afp.com"
            and row["suggested_label"] in {"refuted", "insufficient_evidence"}
            and (row["suggested_label"] == "insufficient_evidence" or (VISUAL_RE.search(row["claim_text"]) and US_RE.search(row["claim_text"] + " " + row["review_title"])))
        )
        or (
            row["publisher_site"] == "snopes.com"
            and row["suggested_label"] == "supported"
            and VISUAL_RE.search(row["claim_text"])
            and US_RE.search(row["claim_text"] + " " + row["review_title"])
        )
    ]
    by_id = {row["candidate_id"]: row for row in candidates}
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(page_record, row) for row in candidates]
        for position, future in enumerate(as_completed(futures), start=1):
            page = future.result()
            source = by_id[page["candidate_id"]]
            page.update(
                {
                    "publisher_site": source["publisher_site"],
                    "review_url": source["review_url"],
                    "claim_text": source["claim_text"],
                    "source_rating": source["textual_rating"],
                    "normalized_label": source["suggested_label"],
                }
            )
            records.append(page)
            if position % 100 == 0:
                print(f"fetched {position}/{len(candidates)} pages", flush=True)

    eligible = [
        row
        for row in records
        if row.get("status") == "ok"
        and row.get("us_signal")
        and normalize_rating(str(row.get("structured_rating", ""))) == row["normalized_label"]
        and float(row.get("claim_similarity", 0.0)) >= 0.50
        and row.get("image_candidates")
        and row.get("evidence_domains")
    ]
    shortlist = select(eligible, quotas)
    output_root = Path(args.output_directory)
    image_root = output_root / "images"
    downloaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(2, args.workers // 2)) as executor:
        futures = [executor.submit(download_first_image, row, image_root) for row in shortlist]
        for future in as_completed(futures):
            downloaded.append(future.result())
    usable = [row for row in downloaded if row.get("image_status") == "ok"]
    usable.sort(key=lambda row: (row["normalized_label"], row["candidate_id"]))
    selected_ids = {row["candidate_id"] for row in usable}

    review_rows: list[dict[str, str]] = []
    for source in candidates:
        row = {field: source.get(field, "") for field in REVIEW_FIELDS}
        page = next((item for item in usable if item["candidate_id"] == source["candidate_id"]), None)
        if page:
            row.update(
                {
                    "decision": "keep",
                    "us_relevant": "yes",
                    "multimodal": "yes",
                    "target_image_verified": "no",
                    "rating_verified": "yes",
                    "claim_image_url": page["claim_image_url"],
                    "claim_time": page["published_at"],
                    "decision_cutoff": page["published_at"],
                    "cutoff_basis": "fact_check_publication",
                    "cutoff_confidence": "confirmed",
                    "normalized_label": page["normalized_label"],
                    "event_id": event_id(page["claim_text"], page["published_at"]),
                    "evidence_query": default_evidence_query(page["claim_text"]),
                    "sufficiency_target": "unlabeled",
                    "reviewer_notes": (
                        "Evidence-first machine screen: structured rating and claim matched; US signal present; "
                        f"downloaded article image; {len(page['evidence_domains'])} exact pre-cutoff Web Archive source domains. "
                        "Visual claim-image confirmation remains pending."
                    ),
                }
            )
        else:
            row.update({"decision": "reject", "exclusion_reason": "outside_evidence_first_buffer_or_machine_gate"})
        review_rows.append(row)

    image_fields = [
        "candidate_id", "status", "source_url", "final_url", "local_path", "content_type",
        "image_sha256", "image_phash", "width", "height", "format", "error",
    ]
    image_rows = [
        {
            "candidate_id": row["candidate_id"],
            "status": "ok",
            "source_url": row["claim_image_url"],
            "final_url": row["claim_image_url"],
            "local_path": row["local_path"],
            "content_type": "image/" + str(row["format"]).casefold(),
            "image_sha256": row["image_sha256"],
            "image_phash": row["image_phash"],
            "width": row["width"],
            "height": row["height"],
            "format": row["format"],
            "error": "",
        }
        for row in usable
    ]
    evidence_fields = [
        "evidence_id", "claim_id", "decision", "exclusion_reason", "url", "domain", "title",
        "gdelt_seen_at", "published_at", "decision_cutoff", "archived_at", "timestamp_confidence",
        "relevance_verified", "publication_time_verified", "relation", "media_type", "source_cluster_id",
        "cluster_verified", "text_sha256", "image_phash", "outgoing_links", "retrieval_query", "reviewer_notes",
    ]
    evidence_rows: list[dict[str, str]] = []
    for row in usable:
        for item in row["evidence"]:
            evidence_id = "ev-" + hashlib.sha256(f"{row['candidate_id']}|{item['original_url']}".encode()).hexdigest()[:20]
            evidence_rows.append(
                {
                    "evidence_id": evidence_id,
                    "claim_id": row["candidate_id"],
                    "decision": "keep",
                    "url": item["original_url"],
                    "domain": item["domain"],
                    "title": item["context"][:300] or item["original_url"],
                    "published_at": item["archived_at"],
                    "decision_cutoff": row["published_at"],
                    "archived_at": item["archived_at"],
                    "timestamp_confidence": "observed",
                    "relevance_verified": "yes",
                    "publication_time_verified": "yes",
                    "relation": "fact_check_cited_source",
                    "media_type": "text",
                    "source_cluster_id": "domain:" + item["domain"],
                    "cluster_verified": "yes",
                    "text_sha256": hashlib.sha256(item["context"].encode()).hexdigest(),
                    "outgoing_links": item["archive_url"],
                    "retrieval_query": default_evidence_query(row["claim_text"]),
                    "reviewer_notes": "Exact Web Archive timestamp embedded in the fact-check citation; original publication time conservatively set to first verified archive observation.",
                }
            )

    write_csv(output_root / "pilot_review_buffer.csv", review_rows, list(REVIEW_FIELDS))
    write_csv(output_root / "image_manifest_buffer.csv", image_rows, image_fields)
    write_csv(output_root / "evidence_review_buffer.csv", evidence_rows, evidence_fields)
    write_csv(
        output_root / "page_audit.csv",
        [
            {
                "candidate_id": row["candidate_id"],
                "status": row.get("status", ""),
                "publisher_site": row.get("publisher_site", ""),
                "normalized_label": row.get("normalized_label", ""),
                "published_at": row.get("published_at", ""),
                "structured_rating": row.get("structured_rating", ""),
                "claim_similarity": row.get("claim_similarity", ""),
                "us_signal": row.get("us_signal", ""),
                "image_candidate_count": len(row.get("image_candidates", [])),
                "evidence_domain_count": len(row.get("evidence_domains", [])),
                "error": row.get("error", ""),
            }
            for row in records
        ],
        [
            "candidate_id", "status", "publisher_site", "normalized_label", "published_at",
            "structured_rating", "claim_similarity", "us_signal", "image_candidate_count",
            "evidence_domain_count", "error",
        ],
    )
    manifest = {
        "candidate_pages": len(candidates),
        "metadata_and_evidence_eligible": len(eligible),
        "selected_before_image_download": len(shortlist),
        "usable_buffer_rows": len(usable),
        "label_counts": {label: sum(row["normalized_label"] == label for row in usable) for label in quotas},
        "claims_with_two_archive_domains": sum(len(row["evidence_domains"]) >= 2 for row in usable),
        "status": "evidence_first_buffer_visual_audit_pending",
    }
    (output_root / "buffer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
