# -*- coding: utf-8 -*-
"""
kb_scrapers.py — Crawls Wikipedia, WHO, FactCheck.org, and PolitiFact
to build the raw JSON knowledge base files consumed by kb_builder.py.

Usage:
    python kb_scrapers.py

Output files (saved to data/):
    auto_fact_base.json   ← Wikipedia
    who_fact_base.json    ← WHO
    factcheck_kb.json     ← FactCheck.org
    politifact_data.json  ← PolitiFact
"""

import json
import re
import time
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

import config

try:
    import wikipedia as _wikipedia_lib
except ImportError:
    _wikipedia_lib = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _clean(text: str, limit: int = 300) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


# ── Wikipedia ───────────────────────────────────────────────────────────────

def scrape_wikipedia(
    topics: List[str]              = config.WIKIPEDIA_TOPICS,
    sentences_per_topic: int       = config.WIKIPEDIA_SENTENCES_PER_TOPIC,
    output_path: str               = config.WIKIPEDIA_KB_PATH,
) -> Dict[str, List[str]]:
    """
    Fetches the Wikipedia page for each topic and extracts the
    first N sentences as facts.
    """
    if _wikipedia_lib is None:
        raise ImportError("Install the wikipedia package:  pip install wikipedia")

    kb: Dict[str, List[str]] = {}
    for topic in topics:
        print(f"  Wikipedia → {topic}")
        try:
            page = _wikipedia_lib.page(topic)
            sentences = re.split(r"(?<=[.!?]) +", page.content)
            kb[topic] = [_clean(s) for s in sentences if len(s) > 50][:sentences_per_topic]
        except Exception as e:
            print(f"    Skipped '{topic}': {e}")
            kb[topic] = []

    with open(output_path, "w") as f:
        json.dump(kb, f, indent=2)
    print(f"Wikipedia KB → {output_path}")
    return kb


# ── WHO ─────────────────────────────────────────────────────────────────────

_WHO_INDEX = "https://www.who.int/news-room/fact-sheets"


def _who_links() -> List[str]:
    res = requests.get(_WHO_INDEX, timeout=15)
    if res.status_code != 200:
        print(f"WHO index fetch failed: {res.status_code}"); return []
    soup = BeautifulSoup(res.content, "lxml")
    seen = set(); links = []
    for a in soup.select("ul li a"):
        href = a.get("href", "")
        if "/fact-sheets/detail/" in href:
            url = "https://www.who.int" + href
            if url not in seen:
                seen.add(url); links.append(url)
    return links


def _who_facts_from_page(url: str, max_facts: int = 5):
    res = requests.get(url, timeout=15)
    if res.status_code != 200: return None, []
    soup = BeautifulSoup(res.content, "lxml")
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"
    facts = []
    section = soup.find("section", {"aria-labelledby": "key-facts"})
    if section:
        facts = [li.get_text(strip=True) for li in section.select("ul li")]
    if not facts:
        for hdr in soup.find_all(["h2", "h3"]):
            if "key facts" in hdr.get_text(strip=True).lower():
                ul = hdr.find_next_sibling("ul")
                if ul:
                    facts = [li.get_text(strip=True) for li in ul.select("li")]
                break
    return title, [f for f in facts if len(f) > 40][:max_facts]


def scrape_who(
    limit: int       = config.WHO_SCRAPE_LIMIT,
    output_path: str = config.WHO_KB_PATH,
) -> None:
    """Crawls WHO fact-sheet pages and extracts key facts."""
    kb: Dict[str, List[str]] = {}
    for url in _who_links()[:limit]:
        title, facts = _who_facts_from_page(url)
        if title and facts:
            kb[title] = facts
            print(f"  WHO → {title[:60]} ({len(facts)} facts)")
        time.sleep(1)
    with open(output_path, "w") as f:
        json.dump(kb, f, indent=2)
    print(f"WHO KB → {output_path}")


# ── FactCheck.org ────────────────────────────────────────────────────────────

_FACTCHECK_BASE = "https://www.factcheck.org"


def _fc_article_links(page: int) -> List[str]:
    res = requests.get(f"{_FACTCHECK_BASE}/page/{page}/", timeout=15)
    if res.status_code != 200: return []
    soup = BeautifulSoup(res.text, "html.parser")
    return [a["href"] for art in soup.find_all("article")
            if (a := art.find("a")) and a.get("href")]


def _fc_parse(url: str):
    res = requests.get(url, timeout=15)
    if res.status_code != 200: return None
    soup = BeautifulSoup(res.text, "html.parser")
    title   = soup.find("h1", class_="entry-title")
    date    = soup.find("time", class_="published")
    content = soup.find("div", class_="entry-content")
    if not (title and date and content): return None
    paras = content.find_all("p")
    return {
        "title":        title.text.strip(),
        "date":         date.get("datetime", ""),
        "claim_summary": paras[0].text.strip() if paras else "",
        "verdict":       paras[-1].text.strip() if paras else "",
        "url":           url,
    }


def scrape_factcheck(
    max_pages: int   = config.FACTCHECK_SCRAPE_PAGES,
    output_path: str = config.FACTCHECK_KB_PATH,
) -> None:
    """Scrapes FactCheck.org and saves claim/verdict pairs."""
    data = []
    for page in range(1, max_pages + 1):
        print(f"  FactCheck.org page {page}/{max_pages}", end="\r")
        for link in _fc_article_links(page):
            item = _fc_parse(link)
            if item: data.append(item)
            time.sleep(0.5)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nFactCheck.org KB → {output_path}  ({len(data)} articles)")


# ── PolitiFact ───────────────────────────────────────────────────────────────

def scrape_politifact(
    page_limit: int  = config.POLITIFACT_SCRAPE_PAGES,
    output_path: str = config.POLITIFACT_KB_PATH,
) -> None:
    """Scrapes PolitiFact fact-check listings."""
    base = "https://www.politifact.com/factchecks/list/?page="
    data = []
    for page in range(1, page_limit + 1):
        print(f"  PolitiFact page {page}/{page_limit}", end="\r")
        try:
            res = requests.get(base + str(page), timeout=15)
            res.raise_for_status()
        except Exception as e:
            print(f"\n  Failed page {page}: {e}"); continue
        soup = BeautifulSoup(res.text, "html.parser")
        for item in soup.find_all("li", class_="o-listicle__item"):
            stmt    = item.find("div", class_="m-statement__quote")
            speaker = item.find("div", class_="m-statement__meta")
            date    = item.find("footer", class_="m-statement__footer")
            rating  = item.find("div", class_="m-statement__meter")
            img     = rating.find("img") if rating else None
            data.append({
                "statement": stmt.text.strip()            if stmt    else None,
                "speaker":   speaker.find("a").text.strip() if speaker and speaker.find("a") else None,
                "date":      date.text.strip()            if date    else None,
                "rating":    img.get("alt")               if img     else None,
            })
        time.sleep(1)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nPolitiFact KB → {output_path}  ({len(data)} items)")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os; os.makedirs(config.DATA_DIR, exist_ok=True)
    print("=== Scraping Wikipedia ===")
    scrape_wikipedia()
    print("\n=== Scraping WHO ===")
    scrape_who()
    print("\n=== Scraping FactCheck.org ===")
    scrape_factcheck()
    print("\n=== Scraping PolitiFact ===")
    scrape_politifact()
    print("\nAll KB sources collected.")
