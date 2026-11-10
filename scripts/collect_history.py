"""Collect every official Lotto 6/45 draw, preserving the old file on failure.

Only a complete, validated sequence from round 1 through the expected latest
round is written. This collector never manufactures draws or substitutes a
partial result. Its fixed verification time is the start of the collection.
"""

from __future__ import annotations

import argparse
import http.client
import ssl
import sys
import time
import urllib.error
from datetime import datetime
from pathlib import Path

if __package__:
    from . import fetch_and_predict as official
else:
    import fetch_and_predict as official


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "draw-history.json"
PAGE_DELAY_SECONDS = 0.15
RETRY_DELAYS_SECONDS = (2.0, 4.0)


def is_transient_failure(error: official.DataUnavailable) -> bool:
    """Retry transport failures, not invalid data or permanent HTTP failures."""
    cause = error.__cause__
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code in (408, 429) or 500 <= cause.code < 600
    if isinstance(cause, urllib.error.URLError):
        return not isinstance(cause.reason, ssl.SSLCertVerificationError)
    return isinstance(cause, (TimeoutError, ConnectionError, OSError, http.client.IncompleteRead)) and not isinstance(
        cause, ssl.SSLCertVerificationError
    )


def fetch_page_with_retry(cursor: int, *, older: bool, now: datetime) -> list[dict]:
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            try:
                return official.fetch_page(cursor, older=older, now=now)
            except http.client.IncompleteRead as exc:
                raise official.DataUnavailable("official response ended prematurely") from exc
        except official.DataUnavailable as exc:
            if isinstance(exc.__cause__, urllib.error.HTTPError):
                exc.__cause__.close()
            if attempt == len(RETRY_DELAYS_SECONDS) or not is_transient_failure(exc):
                raise
            time.sleep(RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("unreachable retry state")


def validate_complete_history(draws: object, now: datetime) -> list[dict]:
    """Validate the full history independently of the recent-report 512 limit."""
    latest = official.latest_expected_round(now)
    if latest < 1 or not isinstance(draws, list) or len(draws) != latest:
        raise official.DataUnavailable("complete history must contain every round from 1")
    clean = [official.validate_draw(draw, now) for draw in draws]
    if [draw["round"] for draw in clean] != list(range(1, latest + 1)):
        raise official.DataUnavailable("history is incomplete, duplicated or out of order")
    return clean


def collect_all_draws(*, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(official.UTC)
    latest = official.latest_expected_round(now)
    if latest < 1:
        raise official.DataUnavailable("no completed draws are expected")
    collected: dict[int, dict] = {}
    cursor, older = latest, False
    # Each accepted page advances at least one round; this also bounds requests.
    for _ in range(latest):
        raw_page = fetch_page_with_retry(cursor, older=older, now=now)
        if not isinstance(raw_page, list) or not raw_page:
            raise official.DataUnavailable("official response has no results")
        page = [official.validate_draw(draw, now) for draw in raw_page]
        rounds = [draw["round"] for draw in page]
        if len(set(rounds)) != len(rounds) or any(r in collected for r in rounds):
            raise official.DataUnavailable("duplicate round in official pagination")
        expected_first = cursor - 1 if older else latest
        if rounds[0] != expected_first:
            raise official.DataUnavailable("latest round missing or pagination did not advance contiguously")
        if rounds != list(range(expected_first, rounds[-1] - 1, -1)):
            raise official.DataUnavailable("official page has missing or unordered rounds")
        collected.update((draw["round"], draw) for draw in page)
        if rounds[-1] == 1:
            return validate_complete_history(
                [collected[r] for r in range(1, latest + 1)], now
            )
        cursor, older = rounds[-1], True
        time.sleep(PAGE_DELAY_SECONDS)
    raise official.DataUnavailable("incomplete official history")


def build_history(draws: list[dict], now: datetime) -> dict:
    clean = validate_complete_history(draws, now)
    stamp = official.utc_timestamp(now)
    latest = clean[-1]
    return {
        "schema_version": 1,
        "status": "current",
        "source": {
            "name": "동행복권",
            "url": official.SOURCE_URL,
            "api_url": official.API_URL,
            "latest_round": latest["round"],
            "latest_draw_date": latest["date"],
            "verified_at": stamp,
        },
        "updated_at": stamp,
        "raw_draws": clean,
        "count": len(clean),
    }


def run(output: Path = OUTPUT_PATH, *, now: datetime | None = None) -> int:
    now = now or datetime.now(official.UTC)
    try:
        report = build_history(collect_all_draws(now=now), now)
        official.atomic_write(output, report)
    except (official.DataUnavailable, OSError) as exc:
        print(f"Official history collection failed; existing output preserved: {exc}", file=sys.stderr)
        return 1
    print(f"Verified {report['count']} official draws; output: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
