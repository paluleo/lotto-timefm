"""Offline collection tests; synthetic fixtures are confined to temporary files."""

import copy
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import call, patch

from scripts import collect_history as history


official = history.official
NOW = datetime(2026, 11, 8, 10, tzinfo=official.UTC)
LATEST = official.latest_expected_round(NOW)
TEST_ROOT = Path(__file__).resolve().parent


def test_only_draw(number):
    return {"round": number, "date": official.round_draw_date(number).isoformat(),
            "numbers": [1, 8, 15, 22, 29, 36], "bonus": 45}


def test_only_pages(latest=LATEST, size=10):
    return [[test_only_draw(r) for r in range(top, max(0, top - size), -1)]
            for top in range(latest, 0, -size)]


def wrapped_transport_error(cause):
    error = official.DataUnavailable("official request failed")
    error.__cause__ = cause
    return error


def test_only_baseline(latest=LATEST - 1):
    stamp = official.FIRST_DRAW + timedelta(weeks=latest - 1, hours=1)
    return history.build_history([test_only_draw(r) for r in range(1, latest + 1)], stamp)


def test_only_recent(now=NOW, count=64):
    stamp = now - timedelta(seconds=60)
    latest = official.latest_expected_round(stamp)
    return official.build_report([test_only_draw(r) for r in range(latest - count + 1, latest + 1)], stamp)


class HistoryCollectionTests(unittest.TestCase):
    def incremental_to_temp(self, cached, recent, *, now=NOW, full_pages=None):
        with tempfile.TemporaryDirectory(prefix=".history-increment-test-", dir=TEST_ROOT) as directory:
            output = Path(directory) / "draw-history.json"
            recent_path = Path(directory) / "recent.json"
            before = cached if isinstance(cached, bytes) else json.dumps(cached).encode() if cached is not None else None
            if before is not None:
                output.write_bytes(before)
            recent_path.write_text(json.dumps(recent), encoding="utf-8")
            with patch.object(official, "fetch_page", side_effect=full_pages or AssertionError("unexpected network call")) as fetch, \
                    patch.object(history.time, "sleep"), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = history.run(output, recent_report=recent_path, now=now)
            raw = output.read_bytes() if output.exists() else None
            return code, raw, before, fetch.call_count

    def run_to_temp(self, pages, *, previous=b"original bytes\n", replace_error=None):
        with tempfile.TemporaryDirectory(prefix=".history-test-", dir=TEST_ROOT) as directory:
            output = Path(directory) / "draw-history.json"
            if previous is not None:
                output.write_bytes(previous)
            with patch.object(official, "fetch_page", side_effect=pages) as fetch, \
                    patch.object(history.time, "sleep") as sleep, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                if replace_error is None:
                    code = history.run(output, now=NOW)
                else:
                    with patch.object(official.os, "replace", side_effect=replace_error):
                        code = history.run(output, now=NOW)
            contents = output.read_bytes() if output.exists() else None
            self.assertEqual(list(Path(directory).glob(".forecast-*.tmp")), [])
            return code, contents, fetch.call_args_list, sleep.call_args_list

    def test_all_1249_rounds_cross_recent_limit_and_use_correct_cursors(self):
        pages = test_only_pages()
        code, raw, fetches, sleeps = self.run_to_temp(pages)
        report = json.loads(raw)
        self.assertEqual(code, 0)
        self.assertGreater(report["count"], official.MAX_DRAWS)
        self.assertEqual(report["count"], LATEST)
        self.assertEqual([d["round"] for d in report["raw_draws"]], list(range(1, LATEST + 1)))
        self.assertEqual(fetches[0], call(LATEST, older=False, now=NOW))
        for index, args in enumerate(fetches[1:], 1):
            self.assertEqual(args, call(pages[index - 1][-1]["round"], older=True, now=NOW))
        self.assertEqual(len(fetches), len(pages))
        self.assertEqual(sleeps, [call(0.15)] * (len(pages) - 1))
        self.assertEqual(report["source"]["latest_round"], LATEST)
        self.assertEqual(report["source"]["verified_at"], official.utc_timestamp(NOW))
        self.assertEqual(report["source"]["api_url"], official.API_URL)
        self.assertEqual(report["updated_at"], report["source"]["verified_at"])
        self.assertEqual(report["status"], "current")
        self.assertEqual(report["source"]["history_verification"], {
            "mode": "full", "full_history_verified_at": official.utc_timestamp(NOW),
            "rechecked_first_round": 1, "rechecked_last_round": LATEST, "baseline_sha256": None})

    def test_incremental_weekly_extension_preserves_full_verification_and_checks_all_overlap(self):
        cached, recent = test_only_baseline(), test_only_recent()
        code, raw, _, requests = self.incremental_to_temp(cached, recent)
        result = json.loads(raw)
        scope = result["source"]["history_verification"]
        self.assertEqual((code, requests, result["count"]), (0, 0, LATEST))
        self.assertEqual(result["raw_draws"], [test_only_draw(r) for r in range(1, LATEST + 1)])
        self.assertEqual(result["source"]["verified_at"], recent["source"]["verified_at"])
        self.assertEqual(result["updated_at"], official.utc_timestamp(NOW))
        self.assertEqual(scope, {
            "mode": "incremental", "full_history_verified_at": cached["source"]["verified_at"],
            "rechecked_first_round": LATEST - 63, "rechecked_last_round": LATEST,
            "baseline_sha256": history.canonical_history_sha256(cached["raw_draws"]),
        })

    def test_legacy_full_cache_migration_and_repeated_increment_preserve_original_time(self):
        cached = test_only_baseline()
        del cached["source"]["history_verification"]
        code, raw, _, requests = self.incremental_to_temp(cached, test_only_recent())
        first = json.loads(raw)
        self.assertEqual((code, requests), (0, 0))
        later = NOW + timedelta(weeks=1)
        code, raw, _, requests = self.incremental_to_temp(first, test_only_recent(later), now=later)
        second = json.loads(raw)
        scope = second["source"]["history_verification"]
        self.assertEqual((code, requests, second["count"]), (0, 0, LATEST + 1))
        self.assertEqual(scope["full_history_verified_at"], cached["source"]["verified_at"])
        self.assertEqual(scope["baseline_sha256"], history.canonical_history_sha256(first["raw_draws"]))

    def test_every_overlapping_draw_including_bonus_must_match(self):
        for offset in (-1, -20, -63):
            cached = test_only_baseline()
            cached["raw_draws"][offset]["bonus"] = 44
            with self.subTest(offset=offset):
                code, raw, before, requests = self.incremental_to_temp(cached, test_only_recent())
                self.assertEqual((code, raw, requests), (1, before, 0))

    def test_overlap_threshold_and_exactly_64_recent_draws(self):
        # 31 overlapping rows are insufficient for a 64-row recent report.
        code, raw, before, requests = self.incremental_to_temp(test_only_baseline(LATEST - 33), test_only_recent())
        self.assertEqual((code, raw, requests), (1, before, 0))
        code, _, _, requests = self.incremental_to_temp(test_only_baseline(LATEST - 32), test_only_recent())
        self.assertEqual((code, requests), (0, 0))
        for count in (20, 32, 63, 65, 512):
            with self.subTest(count=count):
                code, raw, before, requests = self.incremental_to_temp(test_only_baseline(LATEST), test_only_recent(count=count))
                self.assertEqual((code, raw, requests), (1, before, 0))

    def test_cached_incremental_scope_requires_exactly_64_rechecked_draws(self):
        code, raw, _, _ = self.incremental_to_temp(test_only_baseline(), test_only_recent())
        self.assertEqual(code, 0)
        cached = json.loads(raw)
        for count in (20, 32, 63, 65):
            changed = copy.deepcopy(cached)
            changed["source"]["history_verification"]["rechecked_first_round"] = LATEST - count + 1
            with self.subTest(count=count), self.assertRaises(official.DataUnavailable):
                history.validate_cached_history(changed, NOW)

    def test_bad_recent_report_fails_without_full_fetch_or_overwrite(self):
        cases = []
        for mutate in (
            lambda r: r.update(status="stale"),
            lambda r: r.update(schema_version=True),
            lambda r: r["source"].update(api_url="https://example.test/"),
            lambda r: r["predicted_sum"].update(point=0),
            lambda r: r["source"].update(verified_at=official.utc_timestamp(NOW + timedelta(seconds=1))),
        ):
            report = test_only_recent()
            mutate(report)
            cases.append(report)
        cases.append(test_only_recent(NOW - timedelta(minutes=31)))
        for recent in cases:
            with self.subTest(recent_source=recent.get("source")):
                code, raw, before, requests = self.incremental_to_temp(test_only_baseline(), recent)
                self.assertEqual((code, raw, requests), (1, before, 0))

    def test_recent_age_exactly_thirty_minutes_accepted_but_one_second_more_rejected(self):
        at_limit = test_only_recent(NOW - timedelta(minutes=29))
        self.assertIsNotNone(history.validate_recent_report(at_limit, NOW))
        with self.assertRaises(official.DataUnavailable):
            history.validate_recent_report(at_limit, NOW + timedelta(seconds=1))

    def test_recent_before_rollover_cannot_be_merged_after_rollover(self):
        before = datetime(2026, 11, 7, 20, 59, 59, tzinfo=official.KST)
        recent = test_only_recent(before)
        after = before + timedelta(seconds=1)
        with self.assertRaises(official.DataUnavailable):
            history.validate_recent_report(recent, after)

    def test_missing_or_invalid_cache_requires_a_complete_official_refetch(self):
        malformed = test_only_baseline()
        malformed["raw_draws"].pop(10)
        bad_source = test_only_baseline()
        bad_source["source"]["api_url"] = "https://example.test/"
        for cached in (None, b"invalid JSON", malformed, bad_source):
            with self.subTest(cached_type=type(cached).__name__):
                code, raw, _, requests = self.incremental_to_temp(cached, test_only_recent(), full_pages=test_only_pages())
                self.assertEqual((code, requests), (0, len(test_only_pages())))
                self.assertEqual(json.loads(raw)["source"]["history_verification"]["mode"], "full")

    def test_invalid_cache_and_failed_full_refetch_preserve_bytes(self):
        code, raw, before, requests = self.incremental_to_temp(
            b"invalid original cache", test_only_recent(), full_pages=[official.DataUnavailable("official schema error")])
        self.assertEqual((code, raw, requests), (1, before, 1))

    def test_cache_provenance_requires_exact_metadata_and_ordered_timestamps(self):
        for mutate in (
            lambda r: r.update(count=True),
            lambda r: r["source"].update(latest_round=LATEST),
            lambda r: r["source"].update(latest_draw_date="2020-01-01"),
            lambda r: r.update(updated_at="2099-01-01T00:00:00Z"),
            lambda r: r["source"]["history_verification"].update(full_history_verified_at="2099-01-01T00:00:00Z"),
            lambda r: r["source"]["history_verification"].update(rechecked_first_round=2),
            lambda r: r["source"].update(history_verification=None),
        ):
            cached = test_only_baseline()
            mutate(cached)
            with self.subTest(source=cached["source"]), self.assertRaises(official.DataUnavailable):
                history.validate_cached_history(cached, NOW)

    def test_transport_diagnostics_expose_types_and_status_without_error_contents(self):
        secret = "private-token-must-not-be-logged"
        cause = urllib.error.HTTPError(official.API_URL + "?token=" + secret, 429, secret, {}, None)
        error = wrapped_transport_error(cause)
        diagnostic = io.StringIO()
        with patch.object(official, "fetch_page", side_effect=[error, test_only_pages()[0]]), \
                patch.object(history.time, "sleep"), redirect_stderr(diagnostic):
            history.fetch_page_with_retry(LATEST, older=False, now=NOW)
        text = diagnostic.getvalue()
        self.assertIn(f"cursor={LATEST}", text)
        self.assertIn("attempt=1/3", text)
        self.assertIn("cause_type=HTTPError", text)
        self.assertIn("http_status=429", text)
        self.assertNotIn(secret, text)

    def test_missing_duplicate_or_unordered_history_preserves_existing_bytes(self):
        pages = test_only_pages()
        cases = []
        missing_latest = copy.deepcopy(pages)
        missing_latest[0].pop(0)
        cases.append(missing_latest)
        middle_gap = copy.deepcopy(pages)
        middle_gap[0].pop(4)
        cases.append(middle_gap)
        page_gap = copy.deepcopy(pages)
        page_gap[1].pop(0)
        cases.append(page_gap)
        duplicate = copy.deepcopy(pages)
        duplicate[1].insert(0, duplicate[0][-1])
        cases.append(duplicate)
        unordered = copy.deepcopy(pages)
        unordered[0][1], unordered[0][2] = unordered[0][2], unordered[0][1]
        cases.append(unordered)
        missing_one = copy.deepcopy(pages)
        missing_one[-1].pop()
        missing_one.append([])
        cases.append(missing_one)
        cases.append([[]])
        for pages in cases:
            with self.subTest(first_page=pages[0]):
                code, raw, _, _ = self.run_to_temp(pages)
                self.assertEqual(code, 1)
                self.assertEqual(raw, b"original bytes\n")

    def test_invalid_draw_fields_are_not_published_or_retried(self):
        draw = test_only_draw(LATEST)
        for change in ({"date": "2020-01-01"}, {"bonus": 1}, {"bonus": True},
                       {"numbers": [1, 1, 2, 3, 4, 5]}, {"round": True}):
            with self.subTest(change=change):
                code, raw, fetches, sleeps = self.run_to_temp([[{**draw, **change}]])
                self.assertEqual((code, raw), (1, b"original bytes\n"))
                self.assertEqual(len(fetches), 1)
                self.assertEqual(sleeps, [])

    def test_transient_errors_retry_at_two_and_four_seconds_then_succeed(self):
        page = test_only_pages()[0]
        errors = [wrapped_transport_error(urllib.error.URLError("temporary network failure")),
                  wrapped_transport_error(TimeoutError("timeout"))]
        with patch.object(official, "fetch_page", side_effect=[*errors, page]) as fetch, \
                patch.object(history.time, "sleep") as sleep:
            self.assertEqual(history.fetch_page_with_retry(LATEST, older=False, now=NOW), page)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(2.0), call(4.0)])

    def test_exhausted_transient_failure_preserves_existing_file(self):
        error = wrapped_transport_error(urllib.error.HTTPError(official.API_URL, 503, "busy", {}, None))
        code, raw, fetches, sleeps = self.run_to_temp([error, error, error])
        self.assertEqual((code, raw), (1, b"original bytes\n"))
        self.assertEqual(len(fetches), 3)
        self.assertEqual(sleeps, [call(2.0), call(4.0)])

    def test_truncated_http_response_is_retried_and_failure_preserves_file(self):
        error = http.client.IncompleteRead(b"partial response", 100)
        code, raw, fetches, sleeps = self.run_to_temp([error, error, error])
        self.assertEqual((code, raw), (1, b"original bytes\n"))
        self.assertEqual(len(fetches), 3)
        self.assertEqual(sleeps, [call(2.0), call(4.0)])

    def test_permanent_http_and_invalid_json_do_not_retry(self):
        causes = [urllib.error.HTTPError(official.API_URL, 400, "bad request", {}, None),
                  json.JSONDecodeError("bad response", "<html>", 0)]
        errors = [wrapped_transport_error(cause) for cause in causes]
        errors.append(official.DataUnavailable("invalid official row"))
        for error in errors:
            with self.subTest(error=error):
                code, raw, fetches, sleeps = self.run_to_temp([error], previous=None)
                self.assertEqual((code, raw), (1, None))
                self.assertEqual(len(fetches), 1)
                self.assertEqual(sleeps, [])

    def test_atomic_replace_failure_preserves_previous_file_and_cleans_temp(self):
        code, raw, _, _ = self.run_to_temp(test_only_pages(), replace_error=PermissionError("locked"))
        self.assertEqual((code, raw), (1, b"original bytes\n"))

    def test_rollover_changes_required_last_round_and_naive_time_rejected(self):
        before = datetime(2026, 11, 7, 20, 59, 59, tzinfo=official.KST)
        after = before + timedelta(seconds=1)
        draws = [test_only_draw(r) for r in range(1, LATEST)]
        self.assertEqual(len(history.validate_complete_history(draws, before)), LATEST - 1)
        with self.assertRaises(official.DataUnavailable):
            history.validate_complete_history(draws, after)
        with self.assertRaises(ValueError):
            history.validate_complete_history(draws, datetime(2026, 11, 8))


if __name__ == "__main__":
    unittest.main()
