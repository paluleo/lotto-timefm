"""Offline contract tests. Synthetic fixtures exist only here, never in public data."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_and_predict.py"
TEST_ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)

NOW = datetime(2026, 11, 8, 10, tzinfo=pipeline.UTC)
LATEST = pipeline.latest_expected_round(NOW)


def fixture_draws(count=64, latest=LATEST):
    """Intentionally synthetic UNIT TEST data, never passed to a real output file."""
    draws = []
    for number in range(latest - count + 1, latest + 1):
        start = number % 30 + 1
        draws.append({"round": number, "date": pipeline.round_draw_date(number).isoformat(),
                      "numbers": list(range(start, start + 6)), "bonus": 45})
    return draws


def official_payload(draws):
    return {"resultCode": None, "resultMessage": None, "data": {"list": [
        {"ltEpsd": d["round"], "ltRflYmd": d["date"].replace("-", ""),
         "bnsWnNo": d["bonus"], **{f"tm{i + 1}WnNo": n for i, n in enumerate(d["numbers"])}}
        for d in reversed(draws)]}}


class Response(io.BytesIO):
    def __init__(self, payload, url=pipeline.API_URL):
        super().__init__(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        self.url = url

    def geturl(self):
        return self.url


class PipelineTests(unittest.TestCase):
    def run_to_temp(self, previous=None, *, error=None, draws=None, now=NOW):
        with tempfile.TemporaryDirectory(prefix=".pipeline-test-", dir=TEST_ROOT) as directory:
            output = Path(directory) / "forecast.json"
            if previous is not None:
                output.write_text(json.dumps(previous), encoding="utf-8")
            with patch.object(pipeline, "collect_recent_draws", side_effect=error, return_value=draws), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = pipeline.run(output, now=now)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(list(Path(directory).glob(".forecast-*.tmp")), [])
            return code, result

    def test_kst_cutoff_is_saturday_2100(self):
        before = datetime(2026, 11, 7, 20, 59, 59, tzinfo=pipeline.KST)
        after = before + timedelta(seconds=1)
        self.assertEqual(pipeline.latest_expected_round(before), 1248)
        self.assertEqual(pipeline.latest_expected_round(after), 1249)
        self.assertEqual(pipeline.latest_expected_round(after.astimezone(pipeline.UTC)), 1249)

    def test_no_naive_datetime(self):
        with self.assertRaises(ValueError):
            pipeline.latest_expected_round(datetime(2026, 11, 8))

    def test_valid_official_payload(self):
        expected = fixture_draws(20)
        parsed = pipeline.parse_official_page(official_payload(expected), NOW)
        self.assertEqual(parsed, list(reversed(expected)))

    def test_invalid_draw_fields_rejected(self):
        valid = fixture_draws(20)[-1]
        invalid = [
            {**valid, "numbers": [1, 1, 2, 3, 4, 5]},
            {**valid, "numbers": [0, 1, 2, 3, 4, 5]},
            {**valid, "numbers": [1, 2, 3, 4, 5, 46]},
            {**valid, "numbers": [True, 2, 3, 4, 5, 6]},
            {**valid, "numbers": ["1", 2, 3, 4, 5, 6]},
            {**valid, "numbers": [1, 2, 3, 4, 5]},
            {**valid, "bonus": valid["numbers"][0]},
            {**valid, "bonus": None},
            {**valid, "date": "2026-11-06"},
            {**valid, "round": True},
            {**valid, "round": LATEST + 1, "date": pipeline.round_draw_date(LATEST + 1).isoformat()},
        ]
        for draw in invalid:
            with self.subTest(draw=draw), self.assertRaises(pipeline.DataUnavailable):
                pipeline.validate_draw(draw, NOW)

    def test_bad_response_and_duplicates_rejected(self):
        valid = fixture_draws(20)[-1]
        for data in [None, {}, {"data": []}, {"data": {"list": []}},
                     {"resultCode": "ERROR", "data": {"list": []}},
                     official_payload([valid, valid])]:
            with self.subTest(data=data), self.assertRaises(pipeline.DataUnavailable):
                pipeline.parse_official_page(data, NOW)

    def test_http_html_timeout_and_external_redirect_fail_closed(self):
        with patch.object(pipeline.urllib.request, "urlopen", return_value=Response(b"<html>maintenance</html>")):
            with self.assertRaises(pipeline.DataUnavailable):
                pipeline.fetch_page(LATEST, older=False, now=NOW)
        with patch.object(pipeline.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(pipeline.DataUnavailable):
                pipeline.fetch_page(LATEST, older=False, now=NOW)
        with patch.object(pipeline.urllib.request, "urlopen", return_value=Response(
                official_payload(fixture_draws(20)), "https://example.test/result")):
            with self.assertRaises(pipeline.DataUnavailable):
                pipeline.fetch_page(LATEST, older=False, now=NOW)

    def test_endpoint_parameters_and_valid_response(self):
        payload = official_payload(fixture_draws(20))
        with patch.object(pipeline.urllib.request, "urlopen", return_value=Response(payload)) as opener:
            page = pipeline.fetch_page(LATEST, older=False, now=NOW)
        self.assertEqual(page[0]["round"], LATEST)
        self.assertIn("srchDir=center", opener.call_args.args[0].full_url)
        self.assertIn(f"srchLtEpsd={LATEST}", opener.call_args.args[0].full_url)
        with patch.object(pipeline.urllib.request, "urlopen", return_value=Response(payload)) as opener:
            pipeline.fetch_page(LATEST, older=True, now=NOW)
        self.assertIn("srchDir=older", opener.call_args.args[0].full_url)
        self.assertIn(f"srchCursorLtEpsd={LATEST}", opener.call_args.args[0].full_url)

    def test_pagination_requires_contiguous_complete_history(self):
        available = fixture_draws(70)
        pages = [list(reversed(available[max(0, len(available) - end - 10):len(available) - end]))
                 for end in range(0, 70, 10)]
        with patch.object(pipeline, "fetch_page", side_effect=pages) as fetch:
            draws = pipeline.collect_recent_draws(now=NOW)
        self.assertEqual(draws, available[-64:])
        self.assertEqual(fetch.call_count, 7)
        self.assertEqual(fetch.call_args.kwargs["older"], True)
        incomplete = copy.deepcopy(pages)
        incomplete[2].pop(4)
        with patch.object(pipeline, "fetch_page", side_effect=incomplete):
            with self.assertRaises(pipeline.DataUnavailable):
                pipeline.collect_recent_draws(now=NOW)

    def test_missing_latest_and_nonadvancing_page_are_not_success(self):
        old = list(reversed(fixture_draws(20, LATEST - 1)))
        with patch.object(pipeline, "fetch_page", return_value=old):
            with self.assertRaises(pipeline.DataUnavailable):
                pipeline.collect_recent_draws(now=NOW)
        same = list(reversed(fixture_draws(20)[-10:]))
        with patch.object(pipeline, "fetch_page", return_value=same):
            with self.assertRaises(pipeline.DataUnavailable):
                pipeline.collect_recent_draws(now=NOW)

    def test_success_reports_actual_method_and_metadata(self):
        code, result = self.run_to_temp(draws=fixture_draws())
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "current")
        self.assertFalse(result["model"]["timesfm_used"])
        self.assertEqual(result["model"]["id"], "ema-frequency-v1")
        self.assertEqual(result["source"]["latest_round"], LATEST)
        self.assertEqual(result["source"]["latest_draw_date"], "2026-11-07")
        self.assertEqual(result["target_round"], LATEST + 1)
        self.assertEqual(len(result["history_sums"]), 20)
        self.assertEqual(len(result["raw_draws"]), 64)
        self.assertEqual(set(result["ball_weights"]), {str(n) for n in range(1, 46)})
        self.assertEqual(result["history_sums"][-1], sum(fixture_draws()[-1]["numbers"]))

    def test_equal_average_zone_counts_have_stable_order(self):
        draws = fixture_draws(20)
        # Prove width normalization: 20 draws with one occurrence in zones1..4
        # and two in zone5 => final zone ranks first, zones1..4 tie for second.
        for draw in draws:
            draw["numbers"] = [1, 11, 21, 31, 41, 42]
        self.assertEqual(pipeline.report_fields(draws)["hot_zones"], "41~45 · 1~10")

        # Make zone1 and zone2 the equal leaders with differing within-zone
        # distributions; their aggregate counts alone determine the tie.
        for index, draw in enumerate(draws):
            draw["numbers"] = ([1, 2, 11, 12, 21, 31] if index % 2 == 0
                               else [3, 4, 13, 14, 22, 32])
        self.assertEqual(pipeline.report_fields(draws)["hot_zones"], "1~10 · 11~20")

    def test_no_cache_or_legacy_cache_never_creates_numbers(self):
        legacy = {"target_round": 1248, "history_sums": [146] * 20,
                  "model_name": "TimesFM-Compatible EMA Pipeline"}
        for previous in [None, legacy, {"schema_version": 1, "status": []},
                         {"schema_version": 1, "status": {}}, {"schema_version": 1, "status": None}]:
            with self.subTest(previous=previous):
                code, result = self.run_to_temp(previous, error=pipeline.DataUnavailable("offline"))
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "unavailable")
                self.assertIsNone(result["target_round"])
                self.assertIsNone(result["predicted_sum"])
                self.assertEqual(result["history_sums"], [])
                self.assertEqual(result["ball_weights"], {})
                self.assertEqual(result["raw_draws"], [])
                self.assertIsNone(result["source"]["verified_at"])

    def test_failed_refresh_preserves_only_verified_history_as_stale(self):
        previous = pipeline.build_report(fixture_draws(), NOW)
        later = NOW + timedelta(days=8)
        code, result = self.run_to_temp(previous, now=later, error=pipeline.DataUnavailable("offline"))
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["source"]["verified_at"], previous["source"]["verified_at"])
        self.assertEqual(result["updated_at"], pipeline.utc_timestamp(later))
        self.assertEqual(result["raw_draws"], previous["raw_draws"])
        self.assertEqual(result["history_sums"], previous["history_sums"])

    def test_corrupt_or_unproven_cache_rejected(self):
        mutations = [
            lambda d: d.__setitem__("schema_version", True),
            lambda d: d.pop("raw_draws"),
            lambda d: d["raw_draws"][0]["numbers"].__setitem__(0, 99),
            lambda d: d["history_sums"].__setitem__(0, 0),
            lambda d: d["ball_weights"].__setitem__("1", float("nan")),
            lambda d: d["source"].__setitem__("url", "https://example.test/"),
            lambda d: d["source"].__setitem__("verified_at", "2099-01-01T00:00:00Z"),
            lambda d: d["source"].__setitem__("verified_at", "2026-11-06T00:00:00Z"),
            lambda d: d.__setitem__("updated_at", "2026-11-07T12:00:00Z"),
            lambda d: d["model"].__setitem__("timesfm_used", True),
        ]
        for index, mutate in enumerate(mutations):
            previous = pipeline.build_report(fixture_draws(), NOW)
            mutate(previous)
            with self.subTest(mutation=index):
                self.assertIsNone(pipeline.verified_previous_report(previous, NOW))
                self.assertEqual(pipeline.failure_report(previous, NOW)["status"], "unavailable")

    def test_partial_history_cannot_build_report(self):
        cases = [fixture_draws(19), fixture_draws()[:-1], list(reversed(fixture_draws()))]
        for draws in cases:
            with self.subTest(length=len(draws)), self.assertRaises(pipeline.DataUnavailable):
                pipeline.build_report(draws, NOW)

    def test_invalid_previous_file_ignored(self):
        with tempfile.TemporaryDirectory(prefix=".pipeline-test-", dir=TEST_ROOT) as directory:
            output = Path(directory) / "forecast.json"
            output.write_text("{bad json", encoding="utf-8")
            self.assertIsNone(pipeline.read_previous(output))

    def test_failed_atomic_replace_keeps_previous_file(self):
        with tempfile.TemporaryDirectory(prefix=".pipeline-test-", dir=TEST_ROOT) as directory:
            output = Path(directory) / "forecast.json"
            output.write_text("previous", encoding="utf-8")
            with patch.object(pipeline.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    pipeline.atomic_write(output, {"status": "unavailable"})
            self.assertEqual(output.read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(Path(directory).glob(".forecast-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
