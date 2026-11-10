"""Real TimesFM forecasts of 45 weekly occurrence series, never sum forecasts.

The 45 model outputs are relative ranking signals, not calibrated probabilities.
Historical evaluation uses only each target round's preceding draws. Model
failures produce an explicit failed artifact; there is no substitute model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from . import number_statistics as statistics
else:
    import number_statistics as statistics

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"
PACKAGE_VERSION = "2.0.2"
CHECKPOINT_SHA256 = "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8"
CHECKPOINT_BYTES = 925181104
DEFAULT_SEED = 20261110
DEFAULT_CONTEXT = 64
DEFAULT_HOLDOUT = 104
METHODS = ("timesfm", "statistics", "combined", "uniform")
GAMES_PER_METHOD = 5
UTC = timezone.utc
FIRST_DRAW_AT = datetime(2002, 12, 7, 12, tzinfo=UTC)
OFFICIAL_API_URL = "https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do"
ROOT = Path(__file__).resolve().parents[1]


class ModelExecutionError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def timestamp(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return now.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_settings(context: int, holdout: int, seed: int, bootstrap: int) -> None:
    if type(context) is not int or not 32 <= context <= 512 or context % 32:
        raise ValueError("context must be a multiple of 32 between 32 and 512")
    if type(holdout) is not int or not 0 <= holdout <= 512:
        raise ValueError("holdout must be an integer from 0 to 512")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if type(bootstrap) is not int or not 100 <= bootstrap <= 10000:
        raise ValueError("bootstrap resamples must be between 100 and 10000")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def forecast_config(context: int, batch_size: int) -> dict:
    return {"max_context": context, "max_horizon": 128,
            "per_core_batch_size": batch_size, "normalize_inputs": True,
            "use_continuous_quantile_head": True, "force_flip_invariance": True,
            # Preserve negative model signals for relative ranking. These are
            # not probabilities, and clipping them would create artificial ties.
            "infer_is_positive": False, "fix_quantile_crossing": True,
            "return_backcast": False}


def desired_model_metadata(context: int, batch_size: int = 4) -> dict:
    return {"id": MODEL_ID, "name": "TimesFM 2.5 200M (PyTorch)",
            "package": "timesfm", "package_version": PACKAGE_VERSION,
            "revision": MODEL_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES, "checkpoint_verified": False,
            "device": "cpu", "timesfm_used": False, "completed_forecast_calls": 0,
            "point_statistic": "median_0.5_quantile",
            "series": "weekly_main_number_occurrence_0_or_1",
            "forecast_config": forecast_config(context, batch_size),
            "runtime_versions": {"python": platform.python_version()}}


class TimesFMRunner:
    """Pinned safetensors checkpoint and official 2.0.2 CPU API adapter."""

    def __init__(self, context: int = DEFAULT_CONTEXT, *, cache_dir: Path | None = None,
                 batch_size: int = 4, threads: int = 4, local_files_only: bool = False):
        validate_settings(context, 0, DEFAULT_SEED, 100)
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be from 1 to 64")
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("threads must be from 1 to 64")
        self.context = context
        self.cache_dir = cache_dir or ROOT.parent / ".model-cache"
        self.batch_size = batch_size
        self.threads = threads
        self.local_files_only = local_files_only
        self.metadata = desired_model_metadata(context, batch_size)
        self._model = None
        self._numpy = None
        self._torch = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            actual_version = importlib.metadata.version("timesfm")
            if actual_version != PACKAGE_VERSION:
                raise ModelExecutionError("model_load", f"timesfm {PACKAGE_VERSION} is required; found {actual_version}")
            # Set before importing torch. A CPU-only wheel is also recommended.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            import numpy as np
            import torch
            import timesfm
            from huggingface_hub import hf_hub_download

            torch.set_num_threads(self.threads)
            torch.manual_seed(0)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = Path(hf_hub_download(
                repo_id=MODEL_ID, filename="model.safetensors", revision=MODEL_REVISION,
                cache_dir=str(self.cache_dir), local_files_only=self.local_files_only, token=False,
            ))
            if checkpoint.stat().st_size != CHECKPOINT_BYTES or file_sha256(checkpoint) != CHECKPOINT_SHA256:
                raise ModelExecutionError("model_load", "Pinned TimesFM checkpoint checksum mismatch")
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, torch_compile=False,
                cache_dir=str(self.cache_dir), local_files_only=self.local_files_only, token=False,
            )
            if model.model.device.type != "cpu":
                raise ModelExecutionError("model_load", "Expected CPU TimesFM execution")
            model.compile(timesfm.ForecastConfig(**forecast_config(self.context, self.batch_size)))
            self._model, self._numpy, self._torch = model, np, torch
            self.metadata.update({"checkpoint_verified": True, "torch_compile": False,
                                  "cpu_threads": self.threads, "seed": 0,
                                  "runtime_versions": {"python": platform.python_version(),
                                                       "timesfm": actual_version,
                                                       "numpy": np.__version__, "torch": torch.__version__,
                                                       "huggingface_hub": importlib.metadata.version("huggingface-hub"),
                                                       "safetensors": importlib.metadata.version("safetensors")}})
        except ModelExecutionError:
            raise
        except Exception as exc:
            raise ModelExecutionError("model_load", f"TimesFM loading failed ({type(exc).__name__}); no replacement predictions produced") from exc

    def predict(self, contexts: list[list[float]]) -> list[float]:
        if (not isinstance(contexts, list) or not contexts
                or any(not isinstance(values, list) or len(values) != self.context
                       or any(not finite(value) or value not in (0, 1) for value in values)
                       for values in contexts)):
            raise ModelExecutionError("model_inference", "Expected equal-length binary occurrence series")
        self.load()
        try:
            # The upstream forecast method appends padding series to its input
            # list. Always pass a new list; caller-owned contexts stay untouched.
            inputs = [self._numpy.asarray(values, dtype=self._numpy.float32) for values in contexts]
            with self._torch.inference_mode():
                points, _ = self._model.forecast(horizon=1, inputs=inputs)
            if points.shape != (len(contexts), 1):
                raise ModelExecutionError("model_inference", "Unexpected TimesFM one-step output shape")
            result = [float(points[index, 0]) for index in range(len(contexts))]
            if not all(finite(value) for value in result):
                raise ModelExecutionError("model_inference", "TimesFM returned a non-finite score")
            self.metadata["timesfm_used"] = True
            self.metadata["completed_forecast_calls"] += 1
            return result
        except ModelExecutionError:
            raise
        except Exception as exc:
            raise ModelExecutionError("model_inference", f"TimesFM inference failed ({type(exc).__name__}); no replacement predictions produced") from exc


def occurrence_contexts(draws: list[dict], end: int, context: int) -> list[list[float]]:
    if type(end) is not int or not context <= end <= len(draws):
        raise ValueError("target origin lacks the required historical context")
    # end is exclusive: never include the target draw at draws[end].
    history = [set(draw["numbers"]) for draw in draws[end - context:end]]
    return [[float(number in numbers) for numbers in history] for number in range(1, 46)]


def predict_scores(draws: list[dict], end: int, context: int, runner) -> list[float]:
    scores = runner.predict(occurrence_contexts(draws, end, context))
    if not isinstance(scores, list) or len(scores) != 45 or not all(finite(score) for score in scores):
        raise ModelExecutionError("model_inference", "TimesFM must return 45 finite number scores")
    return [float(score) for score in scores]


def method_games(draws: list[dict], target_date: str, scores: list[float], seed: int) -> dict:
    result = {}
    for method in METHODS:
        weights = statistics.recommendation_weights(
            draws, target_date, model_scores=scores,
            method="model" if method == "timesfm" else method,
        )
        games = statistics.generate_games(weights, count=GAMES_PER_METHOD, seed=seed)
        result[method] = {"weights": weights, "games": games}
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired_bootstrap(differences: list[float], *, seed: int, resamples: int) -> dict:
    if not differences or not all(finite(value) for value in differences):
        raise ValueError("paired bootstrap requires finite round-level differences")
    rng = random.Random(seed)
    n = len(differences)
    estimates = [sum(differences[rng.randrange(n)] for _ in range(n)) / n
                 for _ in range(resamples)]
    return {"mean_delta": sum(differences) / n,
            "ci95": [percentile(estimates, .025), percentile(estimates, .975)],
            "unit": "mean_matches_per_game_by_round", "rounds": n}


def evaluate(draws: list[dict], runner, *, context: int = DEFAULT_CONTEXT,
             holdout: int = DEFAULT_HOLDOUT, seed: int = DEFAULT_SEED,
             bootstrap: int = 2000, progress=None) -> dict:
    validate_settings(context, holdout, seed, bootstrap)
    base = {"status": "not_requested" if holdout == 0 else "insufficient",
            "protocol": "rolling-origin-one-step-number-occurrence",
            "context_length": context, "requested_holdout": holdout, "evaluated_count": 0,
            "games_per_method": GAMES_PER_METHOD, "seed": seed,
            "first_target_round": None, "last_target_round": None, "rows": [],
            "metrics": {}, "paired_vs_uniform": {},
            "bootstrap": {"resamples": bootstrap, "seed": seed, "confidence_level": .95,
                          "method": "paired-round-percentile", "resampling_unit": "draw_round"},
            "limitations": [
                "Historical results on the stated fixed holdout do not establish future winning probability.",
                "All model inputs and year/week statistics use only rounds preceding each target.",
                "Five games share a round; bootstrap resamples rounds, not individual games.",
                "Strategies and blend weights were fixed before evaluating this holdout; it is not a tuning set.",
                "Foundation-model pretraining overlap with these public historical records is unknown.",
                "The confidence interval describes the sampled historical mean-match difference, not jackpot odds.",
            ]}
    if holdout == 0 or len(draws) < context + holdout:
        base["reason"] = "holdout_not_requested" if holdout == 0 else "not_enough_draws_for_full_fixed_holdout"
        base["required_draw_count"] = context + holdout
        return base

    rows = []
    for index in range(len(draws) - holdout, len(draws)):
        target = draws[index]
        scores = predict_scores(draws, index, context, runner)
        # Date is a calendar covariate known before the draw; target outcomes
        # are passed only to scoring after all five-game recommendations exist.
        generated = method_games(draws[:index], target["date"], scores, seed + target["round"])
        evaluated = {}
        for method in METHODS:
            games = [{"numbers": numbers, **statistics.score_game(numbers, target)}
                     for numbers in generated[method]["games"]]
            evaluated[method] = {"games": games,
                                 "mean_matches": sum(game["matches"] for game in games) / GAMES_PER_METHOD,
                                 "best_matches": max(game["matches"] for game in games)}
        rows.append({"target_round": target["round"], "target_date": target["date"],
                     "context_start_round": draws[index - context]["round"],
                     "context_end_round": draws[index - 1]["round"],
                     "history_count": index, "seed": seed + target["round"],
                     "actual_numbers": target["numbers"], "actual_bonus": target["bonus"],
                     "raw_scores": {str(n): scores[n - 1] for n in range(1, 46)},
                     "methods": evaluated})
        if progress:
            progress(len(rows), holdout, target["round"])

    metrics, paired = {}, {}
    for method in METHODS:
        games = [game for row in rows for game in row["methods"][method]["games"]]
        rank_counts = {str(rank): 0 for rank in range(1, 6)} | {"none": 0}
        for game in games:
            rank_counts["none" if game["rank"] is None else str(game["rank"])] += 1
        metrics[method] = {"games": len(games),
                           "avg_matches_per_game": sum(game["matches"] for game in games) / len(games),
                           "avg_best_of_five": sum(row["methods"][method]["best_matches"] for row in rows) / len(rows),
                           "prize_rank_counts": rank_counts}
        if method != "uniform":
            differences = [row["methods"][method]["mean_matches"] - row["methods"]["uniform"]["mean_matches"]
                           for row in rows]
            paired[method] = paired_bootstrap(differences, seed=seed, resamples=bootstrap)
    base.update({"status": "complete", "evaluated_count": len(rows),
                 "first_target_round": rows[0]["target_round"], "last_target_round": rows[-1]["target_round"],
                 "rows": rows, "metrics": metrics, "paired_vs_uniform": paired})
    return base


def input_metadata(draws: list[dict]) -> dict:
    canonical = json.dumps(draws, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {"count": len(draws), "first_round": draws[0]["round"], "last_round": draws[-1]["round"],
            "last_draw_date": draws[-1]["date"], "draws_sha256": hashlib.sha256(canonical).hexdigest()}


def expected_latest_round(now: datetime) -> int:
    if now.tzinfo is None:
        raise ValueError("timezone-aware datetime required")
    return max(0, int((now - FIRST_DRAW_AT).total_seconds() // (7 * 86400)) + 1)


def source_metadata(payload: object, draws: list[dict], *, now: datetime | None = None) -> dict | None:
    # A canonical list can be analyzed without claiming verified provenance.
    # A wrapper claiming to be an official current report must satisfy the
    # full source contract; missing metadata must not downgrade it to a list.
    if not isinstance(payload, dict):
        return None
    now = now or datetime.now(UTC)
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported official input schema")
    if payload.get("status") != "current":
        raise ValueError("official input report is not current")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("official input report lacks source metadata")
    if (source.get("name") != "동행복권" or source.get("url") != "https://www.dhlottery.co.kr/"
            or source.get("api_url") != OFFICIAL_API_URL
            or type(source.get("latest_round")) is not int
            or source.get("latest_round") != draws[-1]["round"]
            or source.get("latest_draw_date") != draws[-1]["date"]):
        raise ValueError("input source metadata does not match canonical draws")
    stamp = source.get("verified_at")
    if not isinstance(stamp, str):
        raise ValueError("input report lacks a verification timestamp")
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("input verification timestamp must include timezone")
    updated = payload.get("updated_at")
    if not isinstance(updated, str):
        raise ValueError("official input report lacks update timestamp")
    updated_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))
    if updated_at.tzinfo is None or not parsed <= updated_at <= now:
        raise ValueError("input verification/update timestamps are inconsistent or in the future")
    if (source["latest_round"] != expected_latest_round(parsed)
            or source["latest_round"] != expected_latest_round(now)):
        raise ValueError("official input does not contain the latest expected completed draw")
    return copy.deepcopy(source)


def analyze(payload: object, runner, *, context: int = DEFAULT_CONTEXT,
            holdout: int = DEFAULT_HOLDOUT, seed: int = DEFAULT_SEED,
            bootstrap: int = 2000, progress=None) -> dict:
    validate_settings(context, holdout, seed, bootstrap)
    raw = payload.get("raw_draws") if isinstance(payload, dict) else payload
    draws = statistics.validate_draws(raw)
    if len(draws) < context:
        raise ValueError("not enough draws for the requested model context")
    source = source_metadata(payload, draws)
    target_round = draws[-1]["round"] + 1
    target_date = (date.fromisoformat(draws[-1]["date"]) + timedelta(days=7)).isoformat()
    scores = predict_scores(draws, len(draws), context, runner)
    ranks = statistics.rank_scores(scores)
    recommendations = method_games(draws, target_date, scores, seed + target_round)
    validation = evaluate(draws, runner, context=context, holdout=holdout, seed=seed,
                          bootstrap=bootstrap, progress=progress)
    model = copy.deepcopy(runner.metadata)
    if model.get("timesfm_used") is not True:
        raise ModelExecutionError("model_inference", "Runner did not confirm actual TimesFM inference")
    return {"schema_version": 1, "status": "success", "generated_at": timestamp(),
            "source": source, "input": input_metadata(draws), "model": model,
            "prediction": {"target_round": target_round, "target_date": target_date,
                           "context_length": context, "context_start_round": draws[-context]["round"],
                           "context_end_round": draws[-1]["round"], "horizon": 1,
                           "raw_scores": {str(n): scores[n - 1] for n in range(1, 46)},
                           "rank_scores": {str(n): ranks[n - 1] for n in range(1, 46)},
                           "score_kind": "relative_rank_not_probability"},
            "recommendations": {"games_per_method": GAMES_PER_METHOD, "seed": seed + target_round,
                                "methods": recommendations},
            "statistics": statistics.summarize(draws, target_date),
            "validation": validation, "error": None}


def score_vector(value: object) -> list[float]:
    if not isinstance(value, dict) or set(value) != {str(n) for n in range(1, 46)}:
        raise ModelExecutionError("model_replay", "Replay must contain exactly 45 numbered raw scores")
    values = [value[str(n)] for n in range(1, 46)]
    if not all(finite(v) for v in values):
        raise ModelExecutionError("model_replay", "Replay contains a non-finite raw score")
    return [float(v) for v in values]


def verify_replay_model_candidate(candidate: object, history: list[dict], target_date: str,
                                  scores: list[float], seed: int, actual: dict | None = None) -> None:
    """The model-only sampler is unchanged by the revised historical rule."""
    if not isinstance(candidate, dict):
        raise ModelExecutionError("model_replay", "Missing model-only replay candidates")
    weights = statistics.recommendation_weights(history, target_date, scores, method="model")
    expected_games = statistics.generate_games(weights, count=GAMES_PER_METHOD, seed=seed)
    if actual is None:
        if candidate.get("weights") != weights or candidate.get("games") != expected_games:
            raise ModelExecutionError("model_replay", "Current replay candidates do not match the raw model ranking")
    else:
        expected = [{"numbers": game, **statistics.score_game(game, actual)} for game in expected_games]
        if candidate.get("games") != expected:
            raise ModelExecutionError("model_replay", "Historical replay candidates do not match raw model scores")


class ReplayRunner:
    """Reuse only a hash-authorized real inference artifact on identical inputs.

    The supplied SHA-256 is the trust anchor from the observed original model
    run, not a model attestation derived from JSON metadata alone. No model
    weights are loaded and no replacement predictions are computed here.
    """

    def __init__(self, payload: dict, original_bytes: bytes, expected_sha256: str, *,
                 context: int, holdout: int, seed: int, now: datetime | None = None):
        now = now or datetime.now(UTC)
        actual_sha = hashlib.sha256(original_bytes).hexdigest()
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or actual_sha != expected_sha256.lower()):
            raise ModelExecutionError("model_replay", "Original inference artifact SHA-256 mismatch")
        try:
            saved = json.loads(original_bytes.decode("utf-8-sig"))
            draws = statistics.validate_draws(payload.get("raw_draws")) if isinstance(payload, dict) else []
            source = source_metadata(payload, draws, now=now)
            if source is None:
                raise ValueError("replay requires a verified official current input report")
            if (not isinstance(saved, dict) or type(saved.get("schema_version")) is not int
                    or saved["schema_version"] != 1 or saved.get("status") != "success"
                    or saved.get("error") is not None or "audit_replay" in saved):
                raise ValueError("replay requires the original successful inference artifact")
            if saved.get("source") != source or saved.get("input") != input_metadata(draws):
                raise ValueError("replay input or source differs from the original inference")
            generated = datetime.fromisoformat(saved["generated_at"].replace("Z", "+00:00"))
            verified = datetime.fromisoformat(source["verified_at"].replace("Z", "+00:00"))
            if generated.tzinfo is None or not verified <= generated <= now:
                raise ValueError("invalid original inference timestamp")
            model = saved.get("model")
            if not isinstance(model, dict):
                raise ValueError("missing real model provenance")
            batch_size = model.get("forecast_config", {}).get("per_core_batch_size")
            if type(batch_size) is not int or not 1 <= batch_size <= 64:
                raise ValueError("invalid replay model batch size")
            expected_model = desired_model_metadata(context, batch_size)
            identity_fields = ("id", "name", "package", "package_version", "revision", "checkpoint_sha256",
                               "checkpoint_bytes", "device", "point_statistic", "series", "forecast_config")
            if (any(model.get(key) != expected_model[key] for key in identity_fields)
                    or model.get("checkpoint_verified") is not True or model.get("timesfm_used") is not True
                    or model.get("torch_compile") is not False or model.get("seed") != 0
                    or model.get("runtime_versions", {}).get("timesfm") != PACKAGE_VERSION
                    or model.get("completed_forecast_calls") != holdout + 1):
                raise ValueError("model/checkpoint/configuration is not the pinned real inference protocol")
            prediction = saved.get("prediction", {})
            target_date = (date.fromisoformat(draws[-1]["date"]) + timedelta(days=7)).isoformat()
            target_round = draws[-1]["round"] + 1
            expected_forecast = {"target_round": target_round, "target_date": target_date,
                                 "context_length": context, "context_start_round": draws[-context]["round"],
                                 "context_end_round": draws[-1]["round"], "horizon": 1,
                                 "score_kind": "relative_rank_not_probability"}
            if any(prediction.get(key) != value for key, value in expected_forecast.items()):
                raise ValueError("current replay forecast origin is inconsistent")
            current_scores = score_vector(prediction.get("raw_scores"))
            expected_ranks = {str(n): value for n, value in enumerate(statistics.rank_scores(current_scores), 1)}
            if prediction.get("rank_scores") != expected_ranks:
                raise ValueError("current replay ranks differ from raw scores")
            recommendations = saved.get("recommendations", {})
            if recommendations.get("seed") != seed + target_round or recommendations.get("games_per_method") != GAMES_PER_METHOD:
                raise ValueError("current replay seed or game count is inconsistent")
            verify_replay_model_candidate(recommendations.get("methods", {}).get("timesfm"),
                                          draws, target_date, current_scores, seed + target_round)
            validation = saved.get("validation", {})
            expected_status = "complete" if holdout else "not_requested"
            if (validation.get("status") != expected_status
                    or validation.get("protocol") != "rolling-origin-one-step-number-occurrence"
                    or validation.get("context_length") != context
                    or validation.get("requested_holdout") != holdout
                    or validation.get("evaluated_count") != holdout
                    or validation.get("games_per_method") != GAMES_PER_METHOD
                    or validation.get("seed") != seed):
                raise ValueError("replay holdout protocol differs from the requested fixed protocol")
            rows = validation.get("rows")
            if not isinstance(rows, list) or len(rows) != holdout or len(draws) < context + holdout:
                raise ValueError("replay holdout rows are missing or incomplete")
            if holdout and (validation.get("first_target_round") != draws[-holdout]["round"]
                            or validation.get("last_target_round") != draws[-1]["round"]):
                raise ValueError("replay holdout target range is inconsistent")
            self._calls = [(len(draws), current_scores)]
            ledger = [{"target_round": target_round, "scores": current_scores}]
            for offset, row in enumerate(rows):
                index = len(draws) - holdout + offset
                actual = draws[index]
                expected_row = {"target_round": actual["round"], "target_date": actual["date"],
                                "context_start_round": draws[index - context]["round"],
                                "context_end_round": draws[index - 1]["round"], "history_count": index,
                                "seed": seed + actual["round"], "actual_numbers": actual["numbers"],
                                "actual_bonus": actual["bonus"]}
                if not isinstance(row, dict) or any(row.get(key) != value for key, value in expected_row.items()):
                    raise ValueError("a replay origin or observed target does not match the official input")
                scores = score_vector(row.get("raw_scores"))
                verify_replay_model_candidate(row.get("methods", {}).get("timesfm"), draws[:index],
                                              actual["date"], scores, seed + actual["round"], actual)
                self._calls.append((index, scores))
                ledger.append({"target_round": actual["round"], "scores": scores})
            self._draws, self._context, self._cursor = draws, context, 0
            self.metadata = copy.deepcopy(model)
            score_bytes = json.dumps(ledger, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.audit = {"source_sha256": actual_sha, "original_generated_at": saved["generated_at"],
                          "model_score_ledger_sha256": hashlib.sha256(score_bytes).hexdigest(),
                          "raw_model_predictions_unchanged": True, "inference_rerun": False,
                          "origin_count": len(self._calls), "series_per_origin": 45,
                          "reason": "Recompute the fixed first/second-prize historical selection rule using authorized original model outputs."}
        except ModelExecutionError:
            raise
        except Exception as exc:
            raise ModelExecutionError("model_replay", f"Original inference replay rejected: {exc}") from exc

    def predict(self, contexts: list[list[float]]) -> list[float]:
        if self._cursor >= len(self._calls):
            raise ModelExecutionError("model_replay", "Replay requested an extra inference origin")
        end, scores = self._calls[self._cursor]
        if contexts != occurrence_contexts(self._draws, end, self._context):
            raise ModelExecutionError("model_replay", "Replay binary input tensor or call order differs")
        self._cursor += 1
        return list(scores)

    def assert_exhausted(self) -> None:
        if self._cursor != len(self._calls):
            raise ModelExecutionError("model_replay", "Not every authorized inference origin was replayed")


def failed_report(error: Exception, *, context: int, runner=None) -> dict:
    model = copy.deepcopy(runner.metadata) if runner is not None else desired_model_metadata(context)
    # Keep the actual execution history (a backtest may fail after the current
    # inference succeeded), but never publish partial or replacement forecasts.
    return {"schema_version": 1, "status": "failed", "generated_at": timestamp(),
            "source": None, "input": None, "model": model, "prediction": None,
            "recommendations": None, "statistics": None,
            "validation": {"status": "not_completed", "rows": [], "metrics": {}},
            "error": {"stage": getattr(error, "stage", "input_or_output"),
                      "type": type(error).__name__, "message": str(error)}}


def atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".timesfm-", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(data, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Official raw_draws report or canonical draw list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT, help="0 runs current 45-series inference only")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--replay-model-report", type=Path,
                        help="Reuse an observed real inference artifact; no model download or inference")
    parser.add_argument("--replay-sha256", help="Required trusted SHA-256 of the original inference artifact")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ; source records must be preserved")
    if bool(args.replay_model_report) != bool(args.replay_sha256):
        parser.error("--replay-model-report and --replay-sha256 must be supplied together")
    if args.replay_model_report and args.replay_model_report.resolve() == args.output.resolve():
        parser.error("replay source and output must differ; original inference must be preserved")
    try:
        validate_settings(args.context, args.holdout, args.seed, args.bootstrap)
    except ValueError as exc:
        parser.error(str(exc))
    runner = None
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        if args.replay_model_report:
            runner = ReplayRunner(payload, args.replay_model_report.read_bytes(), args.replay_sha256,
                                  context=args.context, holdout=args.holdout, seed=args.seed)
        else:
            runner = TimesFMRunner(args.context, cache_dir=args.cache_dir, batch_size=args.batch_size,
                                   threads=args.threads, local_files_only=args.local_files_only)
        report = analyze(payload, runner, context=args.context, holdout=args.holdout, seed=args.seed,
                         bootstrap=args.bootstrap,
                         progress=lambda done, total, round_number: print(
                             f"Backtest {done}/{total}: round {round_number}", flush=True))
        if isinstance(runner, ReplayRunner):
            runner.assert_exhausted()
            report["audit_replay"] = runner.audit
        atomic_write(args.output, report)
        print(f"TimesFM number forecast complete: target {report['prediction']['target_round']}; validation {report['validation']['status']}")
        return 0
    except Exception as exc:
        atomic_write(args.output, failed_report(exc, context=args.context, runner=runner))
        print(f"TimesFM failed: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"Cause: {type(exc.__cause__).__name__}: {exc.__cause__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
