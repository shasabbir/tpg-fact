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
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from data.pilot_review import read_review_csv


USER_AGENT = "Mozilla/5.0 (compatible; DTP-Fact/0.1; +mailto:26-94030-1@student.aiub.edu)"
FACT_CHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "factcheck.afp.com",
    "leadstories.com", "reuters.com/fact-check", "apnews.com/ap-fact-check",
}
IGNORED_DOMAINS = {
    "l.join1440.com", "doubleclick.net", "googlesyndication.com",
    "facebook.com/snopes", "x.com/snopes", "twitter.com/snopes",
    "bsky.app", "infosec.exchange",
}
DATE_KEYS = ("datePublished", "uploadDate", "dateCreated")
FIELDS = (
    "evidence_id", "claim_id", "decision", "exclusion_reason", "url", "domain",
    "title", "gdelt_seen_at", "published_at", "decision_cutoff", "archived_at",
    "timestamp_confidence", "relevance_verified", "publication_time_verified",
    "relation", "media_type", "source_cluster_id", "cluster_verified",
    "text_sha256", "image_phash", "outgoing_links", "retrieval_query", "reviewer_notes",
)


def request(url: str, timeout: int = 30) -> tuple[bytes, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    with urlopen(req, timeout=timeout) as response:
        return response.read(), response.geturl()


def domain(url: str) -> str:
    return urlparse(url).netloc.casefold().removeprefix("www.")


def normalize_url(url: str) -> str:
    value = html_module.unescape(url.strip())
    parsed = urlparse(value)
    if parsed.netloc.casefold().endswith("google.com") and parsed.path == "/url":
        value = parse_qs(parsed.query).get("q", [value])[0]
    return unquote(value)


def ignored(url: str) -> bool:
    host = domain(url)
    path_host = host + urlparse(url).path.casefold()
    return (
        not url.startswith(("http://", "https://"))
        or any(host == item or host.endswith("." + item) for item in IGNORED_DOMAINS)
        or any(path_host.startswith(item) for item in FACT_CHECK_DOMAINS)
    )


def jsonld_objects(soup: BeautifulSoup) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for node in soup.find_all("script", type="application/ld+json"):
        value = node.string or node.get_text("", strip=True)
        if not value:
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        queue = parsed if isinstance(parsed, list) else [parsed]
        for item in queue:
            if isinstance(item, dict):
                output.append(item)
                graph = item.get("@graph")
                if isinstance(graph, list):
                    output.extend(child for child in graph if isinstance(child, dict))
    return output


def parse_date(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = date_parser.parse(value)
    except (ValueError, TypeError, OverflowError):
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def page_metadata(url: str) -> dict[str, str]:
    result = {"url": url, "final_url": url, "status": "error", "title": "", "published_at": "", "text_sha256": "", "error": ""}
    try:
        payload, final_url = request(url)
        result["final_url"] = final_url
        soup = BeautifulSoup(payload, "html.parser")
        title_node = soup.find("meta", property="og:title")
        title = str(title_node.get("content", "")).strip() if title_node else ""
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        dates: list[str] = []
        for item in jsonld_objects(soup):
            for key in DATE_KEYS:
                if isinstance(item.get(key), str):
                    dates.append(str(item[key]))
        for attrs in (
            {"property": "article:published_time"}, {"name": "article:published_time"},
            {"itemprop": "datePublished"}, {"name": "date"},
        ):
            node = soup.find("meta", attrs=attrs)
            if node and node.get("content"):
                dates.append(str(node["content"]))
        for node in soup.find_all("time", datetime=True):
            dates.append(str(node["datetime"]))
        published = next((parsed for value in dates if (parsed := parse_date(value))), "")
        text = soup.get_text(" ", strip=True)
        result.update(
            {
                "status": "ok",
                "title": html_module.unescape(title),
                "published_at": published,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    except Exception as error:
        result["error"] = str(error)
    return result


def claim_links(row: dict[str, str], max_links: int) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    try:
        payload, _ = request(row["review_url"])
        soup = BeautifulSoup(payload, "html.parser")
        article = soup.find("article", id="article-content") or soup.find("article")
        if article is None:
            return output
        seen: set[str] = set()
        for anchor in article.find_all("a", href=True):
            url = normalize_url(str(anchor["href"]))
            if ignored(url) or url in seen:
                continue
            seen.add(url)
            parent = anchor.find_parent(["p", "li", "blockquote"])
            context = parent.get_text(" ", strip=True) if parent else anchor.get_text(" ", strip=True)
            output.append(
                {
                    "claim_id": row["candidate_id"],
                    "url": url,
                    "anchor_text": anchor.get_text(" ", strip=True),
                    "citation_context": context[:1200],
                    "decision_cutoff": row["decision_cutoff"],
                    "retrieval_query": row.get("evidence_query", ""),
                }
            )
            if len(output) >= max_links:
                break
    except Exception:
        return output
    return output


def token_set(value: str) -> set[str]:
    stop = {"that", "with", "from", "this", "were", "have", "said", "shown", "shows", "photo", "image", "video", "claim", "president", "united", "states"}
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", value) if len(token) > 3 and token.casefold() not in stop}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect dated sources cited in retained Snopes fact-check articles")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-links-per-claim", type=int, default=12)
    parser.add_argument("--max-evidence-per-claim", type=int, default=4)
    args = parser.parse_args()

    claims = [row for row in read_review_csv(args.review_csv) if row.get("decision", "").casefold() == "keep"]
    claim_by_id = {row["candidate_id"]: row for row in claims}
    citations: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(claim_links, row, args.max_links_per_claim) for row in claims]
        for future in as_completed(futures):
            citations.extend(future.result())

    urls = sorted({item["url"] for item in citations})
    metadata: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(page_metadata, url): url for url in urls}
        for position, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            metadata[futures[future]] = record
            if position % 100 == 0:
                print(f"fetched external pages {position}/{len(urls)}", flush=True)

    by_claim: dict[str, list[dict[str, str]]] = {row["candidate_id"]: [] for row in claims}
    for citation in citations:
        meta = metadata[citation["url"]]
        candidate = {**citation, **meta}
        claim = claim_by_id[citation["claim_id"]]
        cutoff = parse_date(claim["decision_cutoff"])
        published = meta.get("published_at", "")
        if meta.get("status") != "ok" or not published or published > cutoff:
            continue
        claim_tokens = token_set(claim["claim_text"])
        evidence_tokens = token_set(" ".join((citation["citation_context"], meta.get("title", ""))))
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        candidate["relevance_score"] = f"{overlap:.4f}"
        if overlap > 0 or citation["anchor_text"].strip():
            by_claim[citation["claim_id"]].append(candidate)

    rows: list[dict[str, str]] = []
    for claim_id, candidates in by_claim.items():
        candidates.sort(key=lambda item: (float(item["relevance_score"]), item["published_at"]), reverse=True)
        used_domains: set[str] = set()
        for item in candidates:
            host = domain(item["final_url"])
            if not host or host in used_domains:
                continue
            used_domains.add(host)
            evidence_id = "ev-" + hashlib.sha256(f"{claim_id}|{item['final_url']}".encode("utf-8")).hexdigest()[:20]
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "claim_id": claim_id,
                    "decision": "pending_archive_audit",
                    "exclusion_reason": "",
                    "url": item["final_url"],
                    "domain": host,
                    "title": item["title"],
                    "gdelt_seen_at": "",
                    "published_at": item["published_at"],
                    "decision_cutoff": item["decision_cutoff"],
                    "archived_at": "",
                    "timestamp_confidence": "uncertain",
                    "relevance_verified": "yes",
                    "publication_time_verified": "yes",
                    "relation": "fact_check_cited_source",
                    "media_type": "text",
                    "source_cluster_id": "domain:" + host,
                    "cluster_verified": "yes",
                    "text_sha256": item["text_sha256"],
                    "image_phash": "",
                    "outgoing_links": "",
                    "retrieval_query": item["retrieval_query"],
                    "reviewer_notes": (
                        "AI-assisted candidate: cited inside the ClaimReview article; direct page "
                        f"publication metadata predates cutoff; token overlap={item['relevance_score']}. "
                        "Awaiting independent Common Crawl observation."
                    ),
                }
            )
            if len(used_domains) >= args.max_evidence_per_claim:
                break

    output_root = Path(args.output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "evidence_review_prearchive.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "citation_collection_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "claims": len(claims),
                "article_citations": len(citations),
                "unique_external_urls": len(urls),
                "dated_candidates": len(rows),
                "claims_with_dated_candidates": len({row["claim_id"] for row in rows}),
                "status": "awaiting_archive_audit",
                "warning": "Cited-source relevance is AI-assisted and no Common Crawl observation has yet been established.",
            },
            stream,
            indent=2,
        )
    print(json.dumps({"claims": len(claims), "citations": len(citations), "dated_candidates": len(rows), "claims_with_candidates": len({row['claim_id'] for row in rows})}, indent=2))


if __name__ == "__main__":
    main()
