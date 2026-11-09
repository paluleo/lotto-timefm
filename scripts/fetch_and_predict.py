"""Fetch official Lotto 6/45 results and build an explicitly heuristic report.

Only successful, complete responses from the official source produce a current
report. There are no fabricated draws, third-party data fallbacks or TimesFM
imports. The sampler range is a heuristic setting, not a confidence interval.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
UTC = timezone.utc
FIRST_DRAW = datetime(2002, 12, 7, 21, 0, tzinfo=KST)
API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
SOURCE_URL = "https://www.dhlottery.co.kr/"
MODEL = {"id": "ema-frequency-v1", "name": "EMA·출현 빈도 통계", "timesfm_used": False}
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "forecast.json"
MIN_DRAWS = 20
MAX_DRAWS = 512
MAX_RESPONSE_BYTES = 2_000_000


class DataUnavailable(ValueError):
    """The official results cannot be verified completely."""


def utc_timestamp(now: datetime) -> str:
    if now.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return now.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def latest_expected_round(now: datetime) -> int:
    """A round becomes expected at Saturday 21:00 KST, never at midnight."""
    if now.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return max(0, int((now - FIRST_DRAW).total_seconds() // (7 * 86400)) + 1)


def round_draw_date(round_number: int) -> date:
    return FIRST_DRAW.date() + timedelta(weeks=round_number - 1)


def is_integer(value: object) -> bool:
    return type(value) is int


def validate_draw(draw: object, now: datetime) -> dict:
    if not isinstance(draw, dict):
        raise DataUnavailable("draw is not an object")
    round_number = draw.get("round")
    numbers = draw.get("numbers")
    bonus = draw.get("bonus")
    if not is_integer(round_number) or not 1 <= round_number <= latest_expected_round(now):
        raise DataUnavailable("invalid or future round")
    if draw.get("date") != round_draw_date(round_number).isoformat():
        raise DataUnavailable("draw date does not match its round")
    if (not isinstance(numbers, list) or len(numbers) != 6
            or any(not is_integer(n) or not 1 <= n <= 45 for n in numbers)
            or len(set(numbers)) != 6):
        raise DataUnavailable("draw must have six distinct integers from 1 to 45")
    if not is_integer(bonus) or not 1 <= bonus <= 45 or bonus in numbers:
        raise DataUnavailable("invalid bonus number")
    return {"round": round_number, "date": draw["date"],
            "numbers": sorted(numbers), "bonus": bonus}


def parse_official_page(payload: object, now: datetime) -> list[dict]:
    if not isinstance(payload, dict) or "resultCode" not in payload or payload["resultCode"] is not None:
        raise DataUnavailable("official source reported an error")
    data = payload.get("data")
    rows = data.get("list") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise DataUnavailable("official response has no results")
    results = []
    for row in rows:
        if not isinstance(row, dict):
            raise DataUnavailable("invalid official row")
        raw_date = row.get("ltRflYmd")
        if not isinstance(raw_date, str) or len(raw_date) != 8 or not raw_date.isdigit():
            raise DataUnavailable("invalid official draw date")
        draw = {"round": row.get("ltEpsd"),
                "date": f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}",
                "numbers": [row.get(f"tm{i}WnNo") for i in range(1, 7)],
                "bonus": row.get("bnsWnNo")}
        results.append(validate_draw(draw, now))
    if len({d["round"] for d in results}) != len(results):
        raise DataUnavailable("duplicate rounds in official response")
    return sorted(results, key=lambda d: d["round"], reverse=True)


def fetch_page(cursor: int, *, older: bool, now: datetime) -> list[dict]:
    # Endpoint and parameter names are from the official /lt645/result page.
    params = ({"srchDir": "older", "srchCursorLtEpsd": cursor} if older else
              {"srchDir": "center", "srchLtEpsd": cursor})
    request = urllib.request.Request(
        API_URL + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": "Mozilla/5.0 LottoFM/1.0", "Accept": "application/json",
                 "Referer": SOURCE_URL + "lt645/result"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            final_url = urllib.parse.urlsplit(response.geturl())
            if final_url.scheme != "https" or final_url.hostname not in {"www.dhlottery.co.kr", "dhlottery.co.kr"}:
                raise DataUnavailable("official source redirected outside its domain")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise DataUnavailable("official response is too large")
            payload = json.loads(raw.decode("utf-8-sig"))
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataUnavailable(f"official request failed: {type(exc).__name__}") from exc
    return parse_official_page(payload, now)


def validate_sequence(draws: object, now: datetime, *, expected_latest: int | None = None) -> list[dict]:
    if not isinstance(draws, list) or not MIN_DRAWS <= len(draws) <= MAX_DRAWS:
        raise DataUnavailable("not enough verified draw history")
    clean = [validate_draw(draw, now) for draw in draws]
    rounds = [draw["round"] for draw in clean]
    if rounds != list(range(rounds[0], rounds[0] + len(rounds))):
        raise DataUnavailable("draw history is incomplete, duplicated or out of order")
    if expected_latest is not None and rounds[-1] != expected_latest:
        raise DataUnavailable("latest expected draw has not been verified")
    return clean


def collect_recent_draws(count: int = 64, *, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(UTC)
    if not MIN_DRAWS <= count <= MAX_DRAWS:
        raise ValueError(f"count must be between {MIN_DRAWS} and {MAX_DRAWS}")
    latest = latest_expected_round(now)
    if latest < count:
        raise DataUnavailable("not enough completed draws")
    collected: dict[int, dict] = {}
    cursor, older = latest, False
    # Each successful page must advance toward the required lower bound.
    for _ in range(count):
        page = fetch_page(cursor, older=older, now=now)
        if not older and page[0]["round"] != latest:
            raise DataUnavailable("latest expected draw has not been verified")
        if older and page[0]["round"] >= cursor:
            raise DataUnavailable("official pagination did not advance")
        for draw in page:
            if draw["round"] in collected:
                raise DataUnavailable("duplicate draw across official pages")
            collected[draw["round"]] = draw
        oldest = min(collected)
        if oldest <= latest - count + 1:
            wanted = [collected.get(r) for r in range(latest - count + 1, latest + 1)]
            return validate_sequence(wanted, now, expected_latest=latest)
        cursor, older = oldest, True
    raise DataUnavailable("incomplete official pagination")


def report_fields(draws: list[dict]) -> dict:
    """Descriptive statistics and a documented sampler heuristic, not odds."""
    sums = [sum(draw["numbers"]) for draw in draws]
    ema = float(sums[0])
    for value in sums[1:]:
        ema = 0.25 * value + 0.75 * ema
    point = round(0.65 * ema + 0.35 * (sum(sums) / len(sums)), 1)
    # ±12 is an arbitrary generation filter retained from the original design.
    # It has no claimed statistical coverage or predictive performance.
    low, high = max(21, math.floor(point - 12)), min(255, math.ceil(point + 12))
    frequencies = Counter(number for draw in draws for number in draw["numbers"])
    expected_frequency = len(draws) * 6 / 45
    weights = {str(n): round(1 + frequencies[n] / expected_frequency, 4) for n in range(1, 46)}
    odd_counts = Counter(sum(n % 2 for n in d["numbers"]) for d in draws)
    high_counts = Counter(sum(n >= 23 for n in d["numbers"]) for d in draws)
    # Stable tie break, and describe these as observed modes in the UI.
    odd = min(odd_counts, key=lambda n: (-odd_counts[n], n))
    high_count = min(high_counts, key=lambda n: (-high_counts[n], n))
    zones = [("1~10", range(1, 11)), ("11~20", range(11, 21)),
             ("21~30", range(21, 31)), ("31~40", range(31, 41)), ("41~45", range(41, 46))]
    # Compare average occurrence counts exactly. Multiplying by 10/zone width
    # handles the five-number final zone without floating-point tie drift
    # between Python and the browser. Stable sorting keeps the listed order.
    zones.sort(key=lambda z: -sum(frequencies[n] for n in z[1]) * (10 // len(z[1])))
    return {"target_round": draws[-1]["round"] + 1,
            "predicted_sum": {"point": point, "low": low, "high": high},
            "recommended_odd_even": f"홀 {odd} : 짝 {6 - odd}",
            "recommended_high_low": f"저 {6 - high_count} : 고 {high_count}",
            "hot_zones": " · ".join(zone[0] for zone in zones[:2]),
            "history_rounds": [d["round"] for d in draws[-20:]],
            "history_sums": sums[-20:], "ball_weights": weights,
            "sample_count": len(draws)}


def build_report(draws: list[dict], now: datetime) -> dict:
    draws = validate_sequence(draws, now, expected_latest=latest_expected_round(now))
    stamp = utc_timestamp(now)
    return {"schema_version": 1, "status": "current", "updated_at": stamp,
            "source": {"name": "동행복권", "url": SOURCE_URL, "api_url": API_URL,
                       "latest_round": draws[-1]["round"], "latest_draw_date": draws[-1]["date"],
                       "verified_at": stamp},
            "model": dict(MODEL),
            "message": "동행복권 공식 결과로 산출한 참고용 통계입니다. TimesFM은 사용하지 않으며 당첨 확률 향상은 검증되지 않았습니다.",
            **report_fields(draws), "raw_draws": draws}


def parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise DataUnavailable("missing verification timestamp")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataUnavailable("invalid verification timestamp") from exc
    if stamp.tzinfo is None:
        raise DataUnavailable("verification timestamp lacks timezone")
    return stamp


def verified_previous_report(value: object, now: datetime) -> dict | None:
    """Legacy JSON and summaries without source draw records are not a cache."""
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1):
        return None
    if value.get("status") not in ("current", "stale") or value.get("model") != MODEL:
        return None
    try:
        source = value.get("source", {})
        if (source.get("name") != "동행복권" or source.get("url") != SOURCE_URL
                or source.get("api_url") != API_URL):
            return None
        stamp = parse_timestamp(source.get("verified_at"))
        updated = parse_timestamp(value.get("updated_at"))
        if not stamp <= updated <= now:
            return None
        draws = validate_sequence(value.get("raw_draws"), now)
        latest = draws[-1]
        if (source.get("latest_round") != latest["round"]
                or source.get("latest_draw_date") != latest["date"]
                or latest["round"] != latest_expected_round(stamp)):
            return None
        expected = report_fields(draws)
        if any(value.get(key) != item for key, item in expected.items()):
            return None
    except (DataUnavailable, AttributeError, TypeError):
        return None
    return copy.deepcopy(value)


def failure_report(previous: object, now: datetime) -> dict:
    result = verified_previous_report(previous, now)
    if result is not None:
        result["status"] = "stale"
        result["updated_at"] = utc_timestamp(now)
        result["message"] = "공식 데이터 갱신에 실패했습니다. 마지막 검증 자료만 표시하며 새 조합 생성은 중단합니다."
        return result
    return {"schema_version": 1, "status": "unavailable", "updated_at": utc_timestamp(now),
            "source": {"name": "동행복권", "url": SOURCE_URL, "api_url": API_URL,
                       "latest_round": None, "latest_draw_date": None, "verified_at": None},
            "model": dict(MODEL), "target_round": None, "predicted_sum": None,
            "history_rounds": [], "history_sums": [], "ball_weights": {}, "raw_draws": [],
            "message": "검증된 공식 추첨 데이터를 불러오지 못했습니다. 통계와 번호 생성을 중단했습니다."}


def read_previous(path: Path) -> object:
    try:
        if path.stat().st_size > MAX_RESPONSE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def atomic_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".forecast-", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(report, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run(output: Path = OUTPUT_PATH, *, count: int = 64, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    previous = read_previous(output)
    try:
        draws = collect_recent_draws(count, now=now)
        report, exit_code = build_report(draws, now), 0
    except DataUnavailable as exc:
        report, exit_code = failure_report(previous, now), 1
        print(f"Official data unavailable: {exc}", file=sys.stderr)
    atomic_write(output, report)
    print(f"Data status: {report['status']}; output: {output}")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if not MIN_DRAWS <= args.count <= MAX_DRAWS:
        parser.error(f"--count must be between {MIN_DRAWS} and {MAX_DRAWS}")
    return run(args.output, count=args.count)


if __name__ == "__main__":
    raise SystemExit(main())
