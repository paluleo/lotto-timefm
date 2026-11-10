"""Offline logic tests; doubles are explicitly labelled and never published.

These tests do not claim that a real model ran. Actual pinned-checkpoint CPU
inference must be verified separately with the CLI and official source data.
"""

import contextlib
import copy
import hashlib
import io
import json
import math
import random
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scripts import timesfm_forecast as experiment

TEST_ROOT = Path(__file__).resolve().parent


def fixture_draws(count=180, latest=1249):
    rng = random.Random(7432)
    result = []
    for round_number in range(latest - count + 1, latest + 1):
        numbers = rng.sample(range(1, 46), 7)
        result.append({"round": round_number,
                       "date": (date(2002, 12, 7) + timedelta(weeks=round_number - 1)).isoformat(),
                       "numbers": sorted(numbers[:6]), "bonus": numbers[-1]})
    return result


class RecordingDouble:
    """A test-only past-frequency double, never labelled as real TimesFM."""
    def __init__(self):
        self.calls = []
        self.metadata = {"id": "test-double-not-a-real-model", "timesfm_used": True,
                         "checkpoint_verified": False}

    def predict(self, contexts):
        self.calls.append(copy.deepcopy(contexts))
        return [sum(values) / len(values) for values in contexts]


class FakePoints:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = shape or (len(values), 1)

    def __getitem__(self, indexes):
        return self.values[indexes[0]]


class ForecastTests(unittest.TestCase):
    def test_binary_contexts_preserve_number_order_and_exclude_target(self):
        draws = fixture_draws()
        contexts = experiment.occurrence_contexts(draws, 100, 64)
        self.assertEqual(len(contexts), 45)
        self.assertTrue(all(len(values) == 64 for values in contexts))
        for number in range(1, 46):
            self.assertEqual(contexts[number - 1], [float(number in d["numbers"]) for d in draws[36:100]])
        changed = copy.deepcopy(draws)
        changed[100]["numbers"] = [1, 2, 3, 4, 5, 6]
        self.assertEqual(contexts, experiment.occurrence_contexts(changed, 100, 64))

    def test_adapter_uses_fresh_inputs_and_one_step(self):
        runner = experiment.TimesFMRunner(64)
        captured = {}

        class FakeNumpy:
            float32 = object()

            @staticmethod
            def asarray(values, dtype):
                return list(values)

        class FakeModel:
            @staticmethod
            def forecast(horizon, inputs):
                captured["horizon"] = horizon
                captured["count"] = len(inputs)
                inputs += [[0.0] * 64]  # upstream's real list-mutation behavior
                return FakePoints([-0.2 + i / 10 for i in range(45)]), None

        runner._numpy = FakeNumpy
        runner._torch = type("FakeTorch", (), {"inference_mode": staticmethod(contextlib.nullcontext)})
        runner._model = FakeModel()
        contexts = experiment.occurrence_contexts(fixture_draws(), 180, 64)
        original = copy.deepcopy(contexts)
        with patch.object(runner, "load"):
            scores = runner.predict(contexts)
        self.assertEqual(contexts, original)
        self.assertEqual(captured, {"horizon": 1, "count": 45})
        self.assertEqual(len(scores), 45)
        self.assertLess(scores[0], 0)  # raw model signals are not clipped probabilities
        self.assertGreater(scores[-1], 1)
        self.assertTrue(runner.metadata["timesfm_used"])
        self.assertEqual(runner.metadata["completed_forecast_calls"], 1)

    def test_wrong_package_pin_stops_before_model_loading(self):
        runner = experiment.TimesFMRunner()
        with patch.object(experiment.importlib.metadata, "version", return_value="0.0.0"):
            with self.assertRaisesRegex(experiment.ModelExecutionError, "2.0.2 is required"):
                runner.load()
        self.assertFalse(runner.metadata["timesfm_used"])

    def test_finite_score_and_exact_45_contract(self):
        draws = fixture_draws()
        invalid = [[0.1] * 44, [0.1] * 46, [float("nan")] * 45,
                   [float("inf")] * 45, [True] * 45, ["0.1"] * 45]
        for values in invalid:
            runner = RecordingDouble()
            with self.subTest(values=values[:1]), patch.object(runner, "predict", return_value=values):
                with self.assertRaises(experiment.ModelExecutionError):
                    experiment.predict_scores(draws, len(draws), 64, runner)

    def test_every_origin_uses_only_its_past_for_model_and_statistics(self):
        draws = fixture_draws(72)
        runner = RecordingDouble()
        scopes = []
        real_weights = experiment.statistics.recommendation_weights

        def recorded_weights(history, target_date, **kwargs):
            scopes.append((len(history), history[-1]["round"], target_date))
            return real_weights(history, target_date, **kwargs)

        with patch.object(experiment.statistics, "recommendation_weights", side_effect=recorded_weights):
            validation = experiment.evaluate(draws, runner, context=64, holdout=8, bootstrap=100)
        self.assertEqual(len(runner.calls), 8)
        self.assertEqual(len(scopes), 8 * 4)
        for offset, index in enumerate(range(64, 72)):
            self.assertEqual(runner.calls[offset], experiment.occurrence_contexts(draws, index, 64))
            row = validation["rows"][offset]
            self.assertEqual(row["context_end_round"], draws[index]["round"] - 1)
            self.assertEqual(scopes[offset * 4:offset * 4 + 4],
                             [(index, draws[index - 1]["round"], draws[index]["date"])] * 4)

    def test_changing_target_cannot_change_its_recommendations(self):
        original = fixture_draws(72)
        changed = copy.deepcopy(original)
        changed[64]["numbers"], changed[64]["bonus"] = [1, 2, 3, 4, 5, 6], 7
        first = experiment.evaluate(original, RecordingDouble(), holdout=8, bootstrap=100)["rows"][0]
        second = experiment.evaluate(changed, RecordingDouble(), holdout=8, bootstrap=100)["rows"][0]
        self.assertEqual(first["raw_scores"], second["raw_scores"])
        for method in experiment.METHODS:
            self.assertEqual([g["numbers"] for g in first["methods"][method]["games"]],
                             [g["numbers"] for g in second["methods"][method]["games"]])

    def test_fixed_seed_and_full_holdout_metrics(self):
        draws = fixture_draws(72)
        first = experiment.evaluate(draws, RecordingDouble(), holdout=8, seed=91, bootstrap=100)
        second = experiment.evaluate(draws, RecordingDouble(), holdout=8, seed=91, bootstrap=100)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["evaluated_count"], 8)
        for method in experiment.METHODS:
            metrics = first["metrics"][method]
            self.assertEqual(metrics["games"], 40)
            self.assertEqual(sum(metrics["prize_rank_counts"].values()), 40)
            self.assertGreaterEqual(metrics["avg_matches_per_game"], 0)
            self.assertLessEqual(metrics["avg_matches_per_game"], metrics["avg_best_of_five"])
            self.assertLessEqual(metrics["avg_best_of_five"], 6)
            for row in first["rows"]:
                expected = experiment.method_games(draws[:row["history_count"]], row["target_date"],
                                                   [row["raw_scores"][str(n)] for n in range(1, 46)], row["seed"])
                self.assertEqual([g["numbers"] for g in row["methods"][method]["games"]], expected[method]["games"])
        for method in experiment.METHODS[:-1]:
            pair = first["paired_vs_uniform"][method]
            self.assertAlmostEqual(pair["mean_delta"], first["metrics"][method]["avg_matches_per_game"]
                                   - first["metrics"]["uniform"]["avg_matches_per_game"])
            self.assertLessEqual(pair["ci95"][0], pair["ci95"][1])

    def test_insufficient_history_never_silently_reduces_104_holdout(self):
        runner = RecordingDouble()
        result = experiment.evaluate(fixture_draws(64), runner, holdout=104)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["requested_holdout"], 104)
        self.assertEqual(result["evaluated_count"], 0)
        self.assertEqual(result["required_draw_count"], 168)
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["metrics"], {})
        self.assertEqual(runner.calls, [])

    def test_smoke_report_contains_45_rank_signals_and_four_methods(self):
        runner = RecordingDouble()
        report = experiment.analyze(fixture_draws(64), runner, holdout=0, bootstrap=100)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["model"]["id"], "test-double-not-a-real-model")
        self.assertFalse(report["model"]["checkpoint_verified"])
        self.assertEqual(report["validation"]["status"], "not_requested")
        self.assertEqual(report["prediction"]["target_round"], 1250)
        self.assertEqual(set(report["prediction"]["raw_scores"]), {str(n) for n in range(1, 46)})
        self.assertTrue(all(0 <= value <= 1 for value in report["prediction"]["rank_scores"].values()))
        self.assertEqual(report["prediction"]["score_kind"], "relative_rank_not_probability")
        self.assertEqual(set(report["recommendations"]["methods"]), set(experiment.METHODS))
        self.assertIsNone(report["source"])  # test fixture is not invented official provenance
        for method in experiment.METHODS:
            self.assertEqual(len(report["recommendations"]["methods"][method]["games"]), 5)

    def test_bootstrap_is_paired_by_round_and_deterministic(self):
        first = experiment.paired_bootstrap([1.0, 1.0, 1.0], seed=99, resamples=100)
        self.assertEqual(first["mean_delta"], 1)
        self.assertEqual(first["ci95"], [1, 1])
        self.assertEqual(first["rounds"], 3)
        self.assertEqual(first, experiment.paired_bootstrap([1.0, 1.0, 1.0], seed=99, resamples=100))
        for invalid in ([], [float("nan")]):
            with self.assertRaises(ValueError):
                experiment.paired_bootstrap(invalid, seed=99, resamples=100)

    def test_model_failure_never_falls_back_to_statistics(self):
        runner = RecordingDouble()
        error = experiment.ModelExecutionError("model_inference", "test-only model failure")
        with patch.object(runner, "predict", side_effect=error):
            with self.assertRaises(experiment.ModelExecutionError):
                experiment.analyze(fixture_draws(64), runner, holdout=0)
        failed = experiment.failed_report(error, context=64)
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["model"]["timesfm_used"])
        self.assertIsNone(failed["prediction"])
        self.assertIsNone(failed["recommendations"])
        self.assertEqual(failed["validation"]["metrics"], {})

    def test_cli_failure_writes_explicit_artifact_and_preserves_input(self):
        with tempfile.TemporaryDirectory(prefix=".timesfm-test-", dir=TEST_ROOT) as directory:
            input_path, output_path = Path(directory) / "input.json", Path(directory) / "timesfm.json"
            before = json.dumps(fixture_draws(64))
            input_path.write_text(before, encoding="utf-8")
            arguments = ["timesfm_forecast.py", "--input", str(input_path), "--output", str(output_path), "--holdout", "0"]
            with patch.object(experiment.sys, "argv", arguments), \
                    patch.object(experiment, "analyze", side_effect=experiment.ModelExecutionError("model_load", "test-only failure")), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = experiment.main()
            self.assertEqual(code, 1)
            self.assertEqual(input_path.read_text(encoding="utf-8"), before)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["status"], "failed")
            self.assertEqual(list(Path(directory).glob(".timesfm-*.tmp")), [])

    def test_settings_and_future_source_mismatch_rejected(self):
        for context, holdout, seed, bootstrap in [(63, 104, 1, 100), (64, -1, 1, 100),
                                                  (64, 104, True, 100), (64, 104, 1, 0)]:
            with self.assertRaises(ValueError):
                experiment.validate_settings(context, holdout, seed, bootstrap)
        payload = {"raw_draws": fixture_draws(64), "source": {"name": "동행복권", "url": "https://www.dhlottery.co.kr/",
                   "latest_round": 1250, "latest_draw_date": "2026-11-07", "verified_at": "2026-11-10T00:00:00Z"}}
        with self.assertRaises(ValueError):
            experiment.analyze(payload, RecordingDouble(), holdout=0)

    def test_official_wrapper_requires_correct_source_and_fresh_timestamps(self):
        draws = fixture_draws(64)
        now = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
        source = {"name": "동행복권", "url": "https://www.dhlottery.co.kr/",
                  "api_url": experiment.OFFICIAL_API_URL, "latest_round": 1249,
                  "latest_draw_date": "2026-11-07", "verified_at": "2026-11-10T10:00:00Z"}
        payload = {"schema_version": 1, "status": "current", "updated_at": "2026-11-10T10:00:00Z",
                   "source": source, "raw_draws": draws}
        self.assertEqual(experiment.source_metadata(payload, draws, now=now), source)
        mutations = [
            lambda p: p.__setitem__("schema_version", True),
            lambda p: p.__setitem__("status", "stale"),
            lambda p: p.pop("source"),
            lambda p: p["source"].__setitem__("api_url", "https://example.test/data"),
            lambda p: p["source"].__setitem__("verified_at", "2026-11-10T13:00:00Z"),
            lambda p: p.__setitem__("updated_at", "2026-11-10T13:00:00Z"),
            lambda p: p.__setitem__("updated_at", "2026-11-10T09:00:00Z"),
            lambda p: p["source"].__setitem__("verified_at", "2026-11-07T11:59:59Z"),
            lambda p: p["source"].__setitem__("verified_at", "2026-11-10T10:00:00"),
        ]
        for mutate in mutations:
            corrupted = copy.deepcopy(payload)
            mutate(corrupted)
            with self.subTest(corrupted=corrupted.get("source")), self.assertRaises(ValueError):
                experiment.source_metadata(corrupted, draws, now=now)
        # Saturday 21:00 KST makes the prior snapshot stale without changing it.
        with self.assertRaises(ValueError):
            experiment.source_metadata(payload, draws, now=datetime(2026, 11, 14, 12, tzinfo=timezone.utc))
        self.assertIsNone(experiment.source_metadata(draws, draws, now=now))


class ReplayTests(unittest.TestCase):
    """In-memory protocol fixtures only, not evidence that a model really ran."""
    @classmethod
    def setUpClass(cls):
        cls.now = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
        cls.draws = fixture_draws(68)
        cls.payload = {"schema_version": 1, "status": "current", "updated_at": "2026-11-10T10:00:00Z",
                       "raw_draws": cls.draws,
                       "source": {"name": "동행복권", "url": "https://www.dhlottery.co.kr/",
                                  "api_url": experiment.OFFICIAL_API_URL, "latest_round": 1249,
                                  "latest_draw_date": "2026-11-07", "verified_at": "2026-11-10T10:00:00Z"}}
        # Deliberately constructed metadata fixture tests the replay contract;
        # it is kept in memory and never written as a production artifact.
        cls.saved = experiment.analyze(cls.draws, RecordingDouble(), holdout=4, bootstrap=100)
        cls.saved["source"] = copy.deepcopy(cls.payload["source"])
        cls.saved["generated_at"] = "2026-11-10T11:00:00Z"
        cls.saved["model"] = experiment.desired_model_metadata(64)
        cls.saved["model"].update({"timesfm_used": True, "checkpoint_verified": True,
                                  "completed_forecast_calls": 5, "torch_compile": False, "seed": 0,
                                  "runtime_versions": {"timesfm": experiment.PACKAGE_VERSION}})

    def runner(self, saved=None, *, payload=None, expected_sha=None):
        body = json.dumps(saved if saved is not None else self.saved).encode()
        digest = expected_sha or hashlib.sha256(body).hexdigest()
        return experiment.ReplayRunner(payload or self.payload, body, digest,
                                        context=64, holdout=4, seed=experiment.DEFAULT_SEED, now=self.now)

    def test_replay_preserves_all_authorized_vectors_in_exact_context_order(self):
        runner = self.runner()
        vectors = [self.saved["prediction"]["raw_scores"]] + [r["raw_scores"] for r in self.saved["validation"]["rows"]]
        ends = [len(self.draws)] + list(range(len(self.draws) - 4, len(self.draws)))
        for end, vector in zip(ends, vectors):
            output = runner.predict(experiment.occurrence_contexts(self.draws, end, 64))
            self.assertEqual(output, [vector[str(n)] for n in range(1, 46)])
        runner.assert_exhausted()
        self.assertFalse(runner.audit["inference_rerun"])
        self.assertTrue(runner.audit["raw_model_predictions_unchanged"])
        self.assertEqual(runner.audit["origin_count"], 5)
        with self.assertRaises(experiment.ModelExecutionError):
            runner.predict(experiment.occurrence_contexts(self.draws, len(self.draws), 64))

    def test_changed_tensor_wrong_order_or_missing_call_is_rejected(self):
        runner = self.runner()
        wrong = experiment.occurrence_contexts(self.draws, len(self.draws), 64)
        wrong[0][0] = 1.0 - wrong[0][0]
        with self.assertRaises(experiment.ModelExecutionError):
            runner.predict(wrong)
        with self.assertRaises(experiment.ModelExecutionError):
            runner.predict(experiment.occurrence_contexts(self.draws, len(self.draws) - 4, 64))
        with self.assertRaises(experiment.ModelExecutionError):
            runner.assert_exhausted()

    def test_hash_anchor_rejects_score_tampering_even_if_ranks_stay_the_same(self):
        digest = hashlib.sha256(json.dumps(self.saved).encode()).hexdigest()
        changed = copy.deepcopy(self.saved)
        changed["prediction"]["raw_scores"] = {k: value + 0.001 for k, value in changed["prediction"]["raw_scores"].items()}
        with self.assertRaisesRegex(experiment.ModelExecutionError, "SHA-256 mismatch"):
            self.runner(changed, expected_sha=digest)

    def test_mock_older_model_missing_origin_or_inconsistent_candidates_rejected(self):
        mutations = [
            lambda d: d["model"].__setitem__("id", "test-double-not-a-real-model"),
            lambda d: d["model"].__setitem__("package_version", "1.0.0"),
            lambda d: d["model"].__setitem__("revision", "main"),
            lambda d: d["model"].__setitem__("checkpoint_sha256", "0" * 64),
            lambda d: d["model"].__setitem__("checkpoint_verified", False),
            lambda d: d["model"]["forecast_config"].__setitem__("max_context", 128),
            lambda d: d["input"].__setitem__("draws_sha256", "0" * 64),
            lambda d: d["validation"]["rows"].pop(),
            lambda d: d["validation"]["rows"][1].__setitem__("target_round", d["validation"]["rows"][0]["target_round"]),
            lambda d: d["validation"]["rows"][0].__setitem__("context_end_round", 999),
            lambda d: d["prediction"]["raw_scores"].__setitem__("1", float("nan")),
            lambda d: d["prediction"]["rank_scores"].__setitem__("1", -1),
            lambda d: d["recommendations"]["methods"]["timesfm"]["weights"].__setitem__(0, 99),
            lambda d: d["validation"]["rows"][0]["methods"]["timesfm"]["games"][0]["numbers"].__setitem__(0, 99),
            lambda d: d.__setitem__("audit_replay", {}),
        ]
        for mutation in mutations:
            changed = copy.deepcopy(self.saved)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(experiment.ModelExecutionError):
                self.runner(changed)

    def test_replay_requires_verified_current_official_payload(self):
        for mutation in [lambda p: p.__setitem__("status", "stale"),
                         lambda p: p["source"].__setitem__("verified_at", "2026-11-10T13:00:00Z"),
                         lambda p: p["raw_draws"][0].__setitem__("numbers", [1, 2, 3, 4, 5, 6])]:
            changed = copy.deepcopy(self.payload)
            mutation(changed)
            with self.assertRaises(experiment.ModelExecutionError):
                self.runner(payload=changed)

    def test_analyze_reuses_vectors_and_recalculates_games_without_model_load(self):
        runner = self.runner()
        real_source = experiment.source_metadata
        with patch.object(experiment, "source_metadata", side_effect=lambda p, d: real_source(p, d, now=self.now)), \
                patch.object(experiment.TimesFMRunner, "load", side_effect=AssertionError("real model must not be loaded by replay")):
            report = experiment.analyze(self.payload, runner, holdout=4, bootstrap=100)
        runner.assert_exhausted()
        self.assertEqual(report["prediction"]["raw_scores"], self.saved["prediction"]["raw_scores"])
        self.assertEqual([r["raw_scores"] for r in report["validation"]["rows"]],
                         [r["raw_scores"] for r in self.saved["validation"]["rows"]])
        self.assertEqual(report["validation"]["evaluated_count"], 4)
        self.assertEqual(report["statistics"]["historical_rule"]["first_prize_weight"], 0.75)


if __name__ == "__main__":
    unittest.main()
