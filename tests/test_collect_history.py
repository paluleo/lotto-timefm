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


class HistoryCollectionTests(unittest.TestCase):
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
