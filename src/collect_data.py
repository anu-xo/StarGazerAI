from dotenv import load_dotenv
load_dotenv()  
"""
StarSight / StarGazer AI - Day 1: collect GitHub repositories.

Run from the project root (the folder that contains src/ and data/):
    python src/collect_data.py

Needs the GITHUB_TOKEN environment variable (never hard-code the token).
Output: data/raw_repos.csv
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------- Config ----------
LANGUAGES = ["Python", "JavaScript", "Java", "C++", "TypeScript",
             "Go", "C#", "PHP", "Rust", "Kotlin"]
STAR_RANGES = ["10..99", "100..999", ">=1000"]
PAGES_PER_QUERY = 2          # 2 pages x 100 = 200 repos per query
PER_PAGE = 100
SLEEP_SECONDS = 2.5          # keeps us under ~30 requests/minute
OUT_FILE = Path("data/raw_repos.csv")

COLUMNS = ["full_name", "language", "size", "forks_count", "open_issues_count",
           "created_at", "pushed_at", "topics", "license", "description",
           "has_wiki", "has_pages", "stargazers_count"]

URL = "https://api.github.com/search/repositories"


def get_headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set. Set it as an environment variable first.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def extract(item):
    """Pull only the fields we need from one repo in the API response."""
    lic = item.get("license")
    return {
        "full_name": item.get("full_name"),
        "language": item.get("language"),
        "size": item.get("size"),
        "forks_count": item.get("forks_count"),
        "open_issues_count": item.get("open_issues_count"),
        "created_at": item.get("created_at"),
        "pushed_at": item.get("pushed_at"),
        "topics": ",".join(item.get("topics", [])),
        "license": lic.get("spdx_id") if lic else None,
        "description": item.get("description"),
        "has_wiki": item.get("has_wiki"),
        "has_pages": item.get("has_pages"),
        "stargazers_count": item.get("stargazers_count"),
    }


def fetch_page(query, page, headers):
    """One API call, with a simple retry if we hit the rate limit."""
    params = {"q": query, "per_page": PER_PAGE, "page": page}
    for attempt in range(3):
        resp = requests.get(URL, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("items", [])
        if resp.status_code in (403, 429):
            wait = 60
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset and resp.headers.get("X-RateLimit-Remaining") == "0":
                wait = max(int(reset) - int(time.time()), 1) + 2
            print(f"  Rate limited. Waiting {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
            continue
        print(f"  Unexpected status {resp.status_code}: {resp.text[:150]}")
        return []
    return []


def main():
    headers = get_headers()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: remember which queries are already done
    done_file = OUT_FILE.parent / "done_queries.txt"
    done = set(done_file.read_text().splitlines()) if done_file.exists() else set()

    if not OUT_FILE.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(OUT_FILE, index=False)

    queries = [f"language:{lang} stars:{rng}" for lang in LANGUAGES for rng in STAR_RANGES]
    print(f"{len(queries)} queries planned, {len(done)} already done.")

    for i, query in enumerate(queries, start=1):
        if query in done:
            continue
        rows = []
        for page in range(1, PAGES_PER_QUERY + 1):
            items = fetch_page(query, page, headers)
            rows.extend(extract(it) for it in items)
            time.sleep(SLEEP_SECONDS)
            if len(items) < PER_PAGE:
                break

        # Save progress after EVERY query so a crash costs nothing
        if rows:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                OUT_FILE, mode="a", header=False, index=False)
        with open(done_file, "a") as f:
            f.write(query + "\n")
        print(f"[{i}/{len(queries)}] {query}: {len(rows)} repos saved")

    # Remove duplicates by full_name
    df = pd.read_csv(OUT_FILE)
    before = len(df)
    df = df.drop_duplicates(subset="full_name").reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nDone. {before} rows -> {len(df)} after removing duplicates.")


if __name__ == "__main__":
    main()