"""Collect every official Lotto 6/45 draw, preserving the old file on failure.

Only a complete, validated sequence from round 1 through the expected latest
round is written. This collector never manufactures draws or substitutes a
partial result. Its fixed verification time is the start of the collection.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
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
RECENT_MAX_AGE_SECONDS = 30 * 60
RECENT_DRAW_COUNT = 64
MIN_OVERLAP_DRAWS = 32
MAX_HISTORY_FILE_BYTES = 20_000_000


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
            cause = exc.__cause__
            http_status = cause.code if isinstance(cause, urllib.error.HTTPError) else None
            cause_type = type(cause).__name__ if cause is not None else type(exc).__name__
            reason_type = type(cause.reason).__name__ if isinstance(cause, urllib.error.URLError) else "none"
            print(f"Official page failure: cursor={cursor} direction={'older' if older else 'center'} "
                  f"attempt={attempt + 1}/3 cause_type={cause_type} reason_type={reason_type} "
                  f"http_status={http_status}", file=sys.stderr)
            if isinstance(exc.__cause__, urllib.error.HTTPError):
                exc.__cause__.close()
            if attempt == len(RETRY_DELAYS_SECONDS) or not is_transient_failure(exc):
                raise
            time.sleep(RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("unreachable retry state")


def validate_complete_history(draws: object, now: datetime, *, latest: int | None = None) -> list[dict]:
    """Validate the full history independently of the recent-report 512 limit."""
    expected = official.latest_expected_round(now)
    latest = expected if latest is None else latest
    if (type(latest) is not int or not 1 <= latest <= expected
            or not isinstance(draws, list) or len(draws) != latest):
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
            "history_verification": {
                "mode": "full", "full_history_verified_at": stamp,
                "rechecked_first_round": 1, "rechecked_last_round": latest["round"],
                "baseline_sha256": None,
            },
        },
        "updated_at": stamp,
        "raw_draws": clean,
        "count": len(clean),
    }


def read_report(path: Path) -> object:
    try:
        if path.stat().st_size > MAX_HISTORY_FILE_BYTES:
            raise official.DataUnavailable("local report exceeds the size limit")
        raw = path.read_bytes()
        if len(raw) > MAX_HISTORY_FILE_BYTES:
            raise official.DataUnavailable("local report exceeds the size limit")
        return json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise official.DataUnavailable(f"local report cannot be read: {type(exc).__name__}") from exc


def validate_recent_report(payload: object, now: datetime) -> dict:
    """Use only fresh, current official data whose raw draws and summaries agree."""
    if not isinstance(payload, dict) or payload.get("status") != "current":
        raise official.DataUnavailable("recent report must be current")
    verified = official.verified_previous_report(payload, now)
    if verified is None:
        raise official.DataUnavailable("recent official report could not be verified")
    if len(verified["raw_draws"]) != RECENT_DRAW_COUNT:
        raise official.DataUnavailable("recent official report must contain exactly 64 draws")
    source = verified["source"]
    stamp = official.parse_timestamp(source["verified_at"])
    if (source["latest_round"] != official.latest_expected_round(now)
            or (now - stamp).total_seconds() > RECENT_MAX_AGE_SECONDS):
        raise official.DataUnavailable("recent official report is outdated or older than 30 minutes")
    return verified


def validate_cached_history(payload: object, now: datetime) -> tuple[list[dict], str]:
    """Accept a complete earlier baseline, never invent missing historical rows."""
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1 or payload.get("status") != "current"):
        raise official.DataUnavailable("invalid history cache schema or status")
    source = payload.get("source")
    if (not isinstance(source, dict) or source.get("name") != "동행복권"
            or source.get("url") != official.SOURCE_URL or source.get("api_url") != official.API_URL
            or type(source.get("latest_round")) is not int):
        raise official.DataUnavailable("invalid history cache source")
    latest = source["latest_round"]
    clean = validate_complete_history(payload.get("raw_draws"), now, latest=latest)
    if (type(payload.get("count")) is not int or payload["count"] != len(clean)
            or source.get("latest_draw_date") != clean[-1]["date"]):
        raise official.DataUnavailable("history cache metadata disagrees with its draws")
    verified = official.parse_timestamp(source.get("verified_at"))
    updated = official.parse_timestamp(payload.get("updated_at"))
    if (not verified <= updated <= now or official.latest_expected_round(verified) != latest):
        raise official.DataUnavailable("history cache verification time is inconsistent")
    scope = source.get("history_verification")
    # The original full collector wrote no scope metadata. Its validated
    # source.verified_at is the full-fetch baseline for this migration only.
    if "history_verification" not in source:
        return clean, official.utc_timestamp(verified)
    if not isinstance(scope, dict):
        raise official.DataUnavailable("invalid history verification scope")
    full_stamp = official.parse_timestamp(scope.get("full_history_verified_at"))
    first, last = scope.get("rechecked_first_round"), scope.get("rechecked_last_round")
    if (not full_stamp <= verified or official.latest_expected_round(full_stamp) < 1
            or type(first) is not int or type(last) is not int
            or not 1 <= first <= last == latest):
        raise official.DataUnavailable("inconsistent history verification scope")
    if scope.get("mode") == "full":
        if first != 1 or full_stamp != verified or scope.get("baseline_sha256") is not None:
            raise official.DataUnavailable("invalid full-history verification scope")
    elif scope.get("mode") == "incremental":
        digest = scope.get("baseline_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or last - first + 1 != RECENT_DRAW_COUNT):
            raise official.DataUnavailable("invalid incremental history verification scope")
    else:
        raise official.DataUnavailable("unknown history verification mode")
    return clean, official.utc_timestamp(full_stamp)


def canonical_history_sha256(draws: list[dict]) -> str:
    encoded = json.dumps(draws, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def merge_recent_history(cached: list[dict], full_verified_at: str, recent: dict, now: datetime) -> dict:
    """Compare every overlapping row before extending a validated baseline."""
    fresh = recent["raw_draws"]
    first, latest = fresh[0]["round"], fresh[-1]["round"]
    by_round = {draw["round"]: draw for draw in cached}
    overlap = [draw for draw in fresh if draw["round"] in by_round]
    if len(overlap) < MIN_OVERLAP_DRAWS:
        raise official.DataUnavailable("insufficient verified history overlap; run a separate full collection")
    if any(by_round[draw["round"]] != draw for draw in overlap):
        raise official.DataUnavailable("official recent draws conflict with the history cache")
    if cached[-1]["round"] > latest:
        raise official.DataUnavailable("history cache extends beyond the fresh official report")
    if official.parse_timestamp(full_verified_at) > official.parse_timestamp(recent["source"]["verified_at"]):
        raise official.DataUnavailable("recent verification precedes the full-history baseline")
    merged = validate_complete_history([draw for draw in cached if draw["round"] < first] + fresh, now)
    report = build_history(merged, now)
    report["source"]["verified_at"] = recent["source"]["verified_at"]
    report["source"]["history_verification"] = {
        "mode": "incremental", "full_history_verified_at": full_verified_at,
        "rechecked_first_round": first, "rechecked_last_round": latest,
        "baseline_sha256": canonical_history_sha256(cached),
    }
    return report


def run(output: Path = OUTPUT_PATH, *, recent_report: Path | None = None, now: datetime | None = None) -> int:
    now = now or datetime.now(official.UTC)
    try:
        report = None
        if recent_report is not None:
            recent = validate_recent_report(read_report(recent_report), now)
            try:
                cached, full_verified_at = validate_cached_history(read_report(output), now)
            except official.DataUnavailable:
                print("History cache missing or invalid; performing a complete official collection.", file=sys.stderr)
            else:
                # Conflicts and insufficient overlap fail closed. They never
                # silently trigger a replacement collection or overwrite.
                report = merge_recent_history(cached, full_verified_at, recent, now)
        if report is None:
            report = build_history(collect_all_draws(now=now), now)
        official.atomic_write(output, report)
    except (official.DataUnavailable, OSError) as exc:
        print(f"Official history collection failed; existing output preserved: {exc}", file=sys.stderr)
        return 1
    scope = report["source"]["history_verification"]
    print(f"History {scope['mode']}: {report['count']} draws; checked "
          f"{scope['rechecked_first_round']}..{scope['rechecked_last_round']}; output: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--recent-report", type=Path,
                        help="Merge a fresh verified recent official report with an overlapping full-history cache")
    args = parser.parse_args()
    if args.recent_report and args.recent_report.resolve() == args.output.resolve():
        parser.error("recent report and history output must be different files")
    return run(args.output, recent_report=args.recent_report)


if __name__ == "__main__":
    raise SystemExit(main())
