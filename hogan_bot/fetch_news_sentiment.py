"""Fetch BTC news sentiment from CryptoPanic.

Requires a Developer API token from https://cryptopanic.com/developers/api/

Set ``CRYPTOPANIC_KEY`` in your ``.env`` file.

⚠️  QUOTA WARNING — Developer plan limits
------------------------------------------
* 100 requests / month  (≈ 3/day)
* 20 items per page     (fixed by API)
* 24-hour news delay    (not real-time on Developer tier)
* Default: 1 page/run  = 1 request consumed.  Do NOT raise pages > 3
  without upgrading to Growth plan ($199/mo).

Metrics stored daily in ``onchain_metrics``
-------------------------------------------
``news_sentiment_score``
    Net vote ratio: (positive_votes - negative_votes) / total_votes
    Range [-1, 1].  Defaults to 0 when no votes.

``news_volume_norm``
    Story count for the day relative to the 30-day rolling average.
    1.0 = average volume; >1 = above average news flow.

Usage
-----
    # Daily refresh (cron) — uses 1 request
    python -m hogan_bot.fetch_news_sentiment

    # Fetch last N pages (each page = 1 request against your quota)
    python -m hogan_bot.fetch_news_sentiment --pages 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_BASE_URL = "https://cryptopanic.com/api/developer/v2/posts/"
_TIMEOUT = 20
_PAGE_SIZE = 20        # fixed by CryptoPanic API
_QUOTA_WARN_PAGES = 3  # warn if caller requests more than this on Developer plan


def _get_key() -> str:
    key = os.getenv("CRYPTOPANIC_KEY", "").strip()
    if not key:
        sys.exit(
            "CRYPTOPANIC_KEY is not set.\n"
            "1. Register at https://cryptopanic.com/developers/api/\n"
            "2. Add  CRYPTOPANIC_KEY=<your_token>  to your .env"
        )
    return key


def _fetch_page(auth_token: str, page: int = 1) -> dict:
    url = (
        f"{_BASE_URL}?auth_token={auth_token}"
        "&currencies=BTC&kind=news&public=true"
        f"&page={page}"
    )
    for attempt in range(3):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {exc.code} fetching CryptoPanic page {page}") from exc
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Request failed page {page}: {exc}") from exc
            time.sleep(4 ** attempt)
    return {}


def _aggregate_by_day(results: list[dict]) -> dict[str, dict]:
    """Aggregate post votes and counts by UTC date.

    Returns ``{date_str: {pos_votes, neg_votes, total_votes, count}}``.
    """
    by_day: dict[str, dict] = defaultdict(
        lambda: {"pos_votes": 0, "neg_votes": 0, "total_votes": 0, "count": 0}
    )
    for post in results:
        published = post.get("published_at", "")
        if not published:
            continue
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            continue
        date_str = dt.strftime("%Y-%m-%d")
        votes = post.get("votes", {})
        pos = int(votes.get("positive", 0) or 0)
        neg = int(votes.get("negative", 0) or 0)
        by_day[date_str]["pos_votes"] += pos
        by_day[date_str]["neg_votes"] += neg
        by_day[date_str]["total_votes"] += pos + neg
        by_day[date_str]["count"] += 1
    return dict(by_day)


def fetch_and_store(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    pages: int = 1,
) -> int:
    """Fetch recent BTC news from CryptoPanic and store daily sentiment metrics.

    Parameters
    ----------
    pages:
        Number of pages to fetch (20 posts/page = 1 API request each).
        **Developer plan: 100 req/month — keep this at 1 for daily runs.**
        Each page = 1 request against your monthly quota.

    Returns
    -------
    int
        Number of rows written.
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    import sqlite3

    if pages > _QUOTA_WARN_PAGES:
        print(
            f"⚠️  WARNING: pages={pages} will use {pages} requests. "
            f"Developer plan allows only 100/month. Consider pages=1."
        )

    auth_token = _get_key()
    print(f"Fetching CryptoPanic BTC news sentiment ({pages} page(s)) ...")

    all_posts: list[dict] = []
    for page in range(1, pages + 1):
        data = _fetch_page(auth_token, page=page)
        posts = data.get("results", [])
        if not posts:
            break
        all_posts.extend(posts)
        time.sleep(0.5)  # polite pacing

    print(f"  Downloaded {len(all_posts)} posts")
    if not all_posts:
        return 0

    by_day = _aggregate_by_day(all_posts)

    # Compute rolling 30-day average story count for volume normalisation
    # Use existing DB data to inform the baseline
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT value FROM onchain_metrics "
            "WHERE symbol=? AND metric='news_volume_norm' ORDER BY date DESC LIMIT 30",
            (symbol,),
        ).fetchall()
    except Exception:
        rows = []
    conn.close()

    prev_counts = [float(r[0]) for r in rows if r[0]] if rows else []
    # Rough baseline: avg from history or use current batch average
    current_counts = [v["count"] for v in by_day.values() if v["count"] > 0]
    if current_counts:
        count_baseline = float(sum(prev_counts + current_counts) / max(len(prev_counts + current_counts), 1))
    else:
        count_baseline = 20.0

    records: list[tuple[str, str, float]] = []
    for date_str, agg in sorted(by_day.items()):
        total = agg["total_votes"]
        if total > 0:
            sent = (agg["pos_votes"] - agg["neg_votes"]) / total
        else:
            sent = 0.0
        vol_norm = agg["count"] / max(count_baseline, 1.0)
        records.append((date_str, "news_sentiment_score", round(sent, 6)))
        records.append((date_str, "news_volume_norm", round(vol_norm, 6)))
        print(f"  {date_str}  sentiment={sent:+.3f}  vol_norm={vol_norm:.2f}  ({agg['count']} stories)")

    conn = get_connection(db_path)
    written = upsert_onchain(conn, symbol, records)
    conn.close()
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch BTC news sentiment from CryptoPanic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Symbol for DB storage")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    p.add_argument("--pages", type=int, default=1, help="Pages to fetch (20 posts/page = 1 request each; Developer plan: 100 req/mo)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fetch_and_store(symbol=args.symbol, db_path=args.db, pages=args.pages)


if __name__ == "__main__":
    main()
