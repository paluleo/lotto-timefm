"""Historical number statistics and reproducible candidate combinations.

These statistics describe official draws, not winning ticket purchase records.
The six second-prize combinations for a draw are derived from five main numbers
and the bonus. Model ranks and sampling weights are not winning probabilities.
Every caller must supply only observations earlier than its target date.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from datetime import date, timedelta

FIRST_DRAW_DATE = date(2002, 12, 7)
NUMBER_COUNT = 45
MAIN_COUNT = 6
PRIOR_STRENGTH = 20.0
FIRST_PRIZE_BLEND = 0.75
SECOND_PRIZE_BLEND = 0.25
MIN_WEIGHT = 0.01
MAX_HISTORY = 10_000


def _date(value: object) -> date:
    if isinstance(value, date) and type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError("date must be an ISO date string or datetime.date")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid ISO date") from exc
    if result.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD")
    return result


def _numbers(value: object) -> list[int]:
    if (not isinstance(value, (list, tuple)) or len(value) != MAIN_COUNT
            or any(type(n) is not int or not 1 <= n <= NUMBER_COUNT for n in value)
            or len(set(value)) != MAIN_COUNT):
        raise ValueError("six distinct integer numbers from 1 to 45 are required")
    return sorted(value)


def _draw(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("draw must be an object")
    round_number = value.get("round")
    if type(round_number) is not int or not 1 <= round_number <= MAX_HISTORY:
        raise ValueError("invalid draw round")
    draw_date = _date(value.get("date"))
    if draw_date != FIRST_DRAW_DATE + timedelta(weeks=round_number - 1):
        raise ValueError("draw date does not match its round")
    numbers = _numbers(value.get("numbers"))
    bonus = value.get("bonus")
    if type(bonus) is not int or not 1 <= bonus <= NUMBER_COUNT or bonus in numbers:
        raise ValueError("bonus must be a distinct number from 1 to 45")
    return {"round": round_number, "date": draw_date.isoformat(),
            "numbers": numbers, "bonus": bonus}


def validate_draws(draws: object) -> list[dict]:
    """Return a fresh, canonical, contiguous history; never synthesize a gap."""
    if not isinstance(draws, (list, tuple)) or not 1 <= len(draws) <= MAX_HISTORY:
        raise ValueError("a nonempty draw history is required")
    clean = [_draw(draw) for draw in draws]
    rounds = [draw["round"] for draw in clean]
    if rounds != list(range(rounds[0], rounds[0] + len(rounds))):
        raise ValueError("history must contain consecutive rounds in ascending order")
    return clean


def _history_for_target(draws: object, target_date: object) -> tuple[list[dict], date]:
    clean = validate_draws(draws)
    target = _date(target_date)
    if target.weekday() != 5:
        raise ValueError("target must be a Saturday draw date")
    if _date(clean[-1]["date"]) >= target:
        raise ValueError("target or future draws cannot be used as history")
    return clean, target


def second_prize_combinations(draw: dict) -> list[list[int]]:
    """Six qualifying combinations, not six observed winning purchases."""
    clean = _draw(draw)
    main = clean["numbers"]
    return sorted(sorted(main[:i] + main[i + 1:] + [clean["bonus"]])
                  for i in range(MAIN_COUNT))


def rank_scores(values: object) -> list[float]:
    """Average tied ranks scaled to 0..1; an all-tied vector maps to 0.5."""
    if (not isinstance(values, (list, tuple)) or len(values) != NUMBER_COUNT
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("45 finite numeric model scores are required")
    order = sorted(range(NUMBER_COUNT), key=lambda i: values[i])
    result = [0.0] * NUMBER_COUNT
    start = 0
    while start < NUMBER_COUNT:
        end = start + 1
        while end < NUMBER_COUNT and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2 / (NUMBER_COUNT - 1)
        for i in order[start:end]:
            result[i] = average_rank
        start = end
    return result


def _first_second_rates(draws: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Smoothed inclusion rates using a prior measured in actual draws.

Each main number occurs in five of the six derived second-prize combinations;
the bonus occurs in all six. Divide those counts by six before adding the
20-draw prior. Six derived combinations are not six independent observations.
The 75:25 mixture is a predeclared experimental rule, not a fitted probability.
"""
    main = Counter(number for draw in draws for number in draw["numbers"])
    bonus = Counter(draw["bonus"] for draw in draws)
    prior = PRIOR_STRENGTH * MAIN_COUNT / NUMBER_COUNT
    denominator = len(draws) + PRIOR_STRENGTH
    first = [(main[n] + prior) / denominator for n in range(1, NUMBER_COUNT + 1)]
    second = [((5 * main[n] + 6 * bonus[n]) / 6 + prior) / denominator
              for n in range(1, NUMBER_COUNT + 1)]
    mixed = [FIRST_PRIZE_BLEND * f + SECOND_PRIZE_BLEND * s
             for f, s in zip(first, second)]
    return first, second, mixed


def _main_rates(draws: list[dict]) -> list[float]:
    return _first_second_rates(draws)[0]


def _historical_rates(draws: list[dict]) -> list[float]:
    return _first_second_rates(draws)[2]


def recommendation_weights(draws: object, target_date: object,
                           model_scores: object = None,
                           method: str = "combined") -> list[float]:
    """Fixed experimental rule; coefficients must not be fitted on test targets.

combined = 50% model rank + 30% all-history rank + 20% same-ISO-week rank.
statistics = 60% all-history rank + 40% same-ISO-week rank.
Each historical rank uses 75% first-prize inclusion + 25% derived second-prize
inclusion, smoothed with a 20-actual-draw prior. Derived combinations are not
independent draws. These weights are fixed before inspecting test performance.
The positive floor enables sampling every number. This is not a probability
calibration, and the exact generated combination has no claimed win likelihood.
"""
    clean, target = _history_for_target(draws, target_date)
    if method not in {"combined", "statistics", "model", "uniform"}:
        raise ValueError("unknown recommendation method")
    if method == "uniform":
        return [1.0] * NUMBER_COUNT
    overall = rank_scores(_historical_rates(clean))
    week = target.isocalendar().week
    seasonal = rank_scores(_historical_rates([d for d in clean
                                              if _date(d["date"]).isocalendar().week == week]))
    if method == "statistics":
        scores = [0.6 * a + 0.4 * s for a, s in zip(overall, seasonal)]
    else:
        model = rank_scores(model_scores)
        scores = model if method == "model" else [
            0.5 * m + 0.3 * a + 0.2 * s for m, a, s in zip(model, overall, seasonal)]
    return [max(MIN_WEIGHT, value) for value in scores]


def generate_games(weights: object, count: int = 5, seed: int = 0) -> list[list[int]]:
    """Seeded weighted sampling without replacement, unique across games too."""
    if (not isinstance(weights, (list, tuple)) or len(weights) != NUMBER_COUNT
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0
                   for w in weights)):
        raise ValueError("45 finite positive weights are required")
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("game count must be between 1 and 100")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    # Scaling avoids overflow when accepting otherwise valid large weights.
    maximum = max(weights)
    scaled = [float(w / maximum) for w in weights]
    if any(w == 0 for w in scaled):
        raise ValueError("weight ratio is too extreme for reliable sampling")
    rng = random.Random(seed)
    games: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(max(1000, count * 100)):
        available = list(range(NUMBER_COUNT))
        chosen = []
        for _ in range(MAIN_COUNT):
            threshold = rng.random() * math.fsum(scaled[i] for i in available)
            selected = available[-1]
            for i in available:
                threshold -= scaled[i]
                if threshold < 0:
                    selected = i
                    break
            chosen.append(selected + 1)
            available.remove(selected)
        game = tuple(sorted(chosen))
        if game not in seen:
            seen.add(game)
            games.append(list(game))
            if len(games) == count:
                return games
    raise ValueError("could not produce enough distinct games with these weights")


def score_game(numbers: object, actual: dict) -> dict:
    """Apply rank conditions to one candidate; no payout or odds inference."""
    candidate = _numbers(numbers)
    draw = _draw(actual)
    matches = len(set(candidate) & set(draw["numbers"]))
    bonus_match = draw["bonus"] in candidate
    rank = (1 if matches == 6 else 2 if matches == 5 and bonus_match else
            3 if matches == 5 else 4 if matches == 4 else 5 if matches == 3 else None)
    return {"matches": matches, "bonus_match": bonus_match, "rank": rank}


def _combination_counts(draws: list[dict], second: bool = False) -> Counter:
    values = (tuple(combo) for draw in draws for combo in second_prize_combinations(draw)) \
        if second else (tuple(draw["numbers"]) for draw in draws)
    return Counter(values)


def _repeat_summary(counts: Counter, limit: int = 10) -> dict:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    preview = ordered[:limit]
    cutoff = preview[-1][1] if preview else None
    return {"observations": sum(counts.values()), "unique_combinations": len(counts),
            "repeated_combinations": sum(n > 1 for n in counts.values()),
            "maximum_count": max(counts.values(), default=0),
            "preview": [{"numbers": list(combo), "count": n} for combo, n in preview],
            "preview_limit": limit, "preview_cutoff_count": cutoff,
            "total_tied_at_cutoff": sum(n == cutoff for n in counts.values()) if cutoff else 0,
            "tie_break": "lexicographic display order only; not predictive preference"}


def _group(draws: list[dict]) -> dict:
    count = len(draws)
    main = Counter(n for d in draws for n in d["numbers"])
    bonus = Counter(d["bonus"] for d in draws)
    first_smoothed, second_smoothed, historical = _first_second_rates(draws)
    return {"sample_count": count, "rate_denominator": "draws in this group",
            "second_qualifying_rate_denominator": "six derived combinations per actual draw",
            "numbers": [{"number": n, "main_hits": main[n], "bonus_hits": bonus[n],
                         "main_rate": main[n] / count if count else None,
                         "bonus_rate": bonus[n] / count if count else None,
                         "second_qualifying_hits": 5 * main[n] + 6 * bonus[n],
                         "second_qualifying_rate": (5 * main[n] + 6 * bonus[n]) / (6 * count)
                         if count else None,
                         "smoothed_main_rate": first_smoothed[n - 1],
                         "smoothed_second_qualifying_rate": second_smoothed[n - 1],
                         "historical_selection_score": historical[n - 1]}
                        for n in range(1, NUMBER_COUNT + 1)]}


def summarize(draws: object, target_date: object) -> dict:
    """Exact annual/week counts with sample sizes and honest tied previews."""
    clean, target = _history_for_target(draws, target_date)
    years: dict[int, list[dict]] = defaultdict(list)
    weeks: dict[int, list[dict]] = defaultdict(list)
    year_weeks: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for draw in clean:
        iso = _date(draw["date"]).isocalendar()
        years[iso.year].append(draw)
        weeks[iso.week].append(draw)
        year_weeks[(iso.year, iso.week)].append(draw)
    iso = target.isocalendar()
    target_history = weeks.get(iso.week, [])
    compact = []
    for (year, week), group in sorted(year_weeks.items()):
        compact.append({"iso_year": year, "iso_week": week, "sample_count": len(group),
                        "rounds": [d["round"] for d in group],
                        "main_counts": {str(n): c for n, c in sorted(Counter(
                            n for d in group for n in d["numbers"]).items())},
                        "bonus_counts": {str(n): c for n, c in sorted(Counter(
                            d["bonus"] for d in group).items())},
                        "first_prize_combinations": [
                            {"numbers": list(ns), "count": c}
                            for ns, c in sorted(_combination_counts(group).items())],
                        "second_prize_qualifying_combinations": [
                            {"numbers": list(ns), "count": c}
                            for ns, c in sorted(_combination_counts(group, second=True).items())]})
    return {"sample_count": len(clean), "first_round": clean[0]["round"],
            "last_round": clean[-1]["round"], "first_date": clean[0]["date"],
            "last_date": clean[-1]["date"], "target_date": target.isoformat(),
            "target_iso_year": iso.year, "target_iso_week": iso.week,
            "week_definition": "ISO-8601 week/year of the official Saturday draw date",
            "second_prize_definition": "Six derived qualifying combinations per draw: five main numbers plus bonus. These are not winning ticket purchase records.",
            "smoothing": {"prior_strength_draws": PRIOR_STRENGTH,
                          "main_rate_prior": MAIN_COUNT / NUMBER_COUNT},
            "historical_rule": {
                "id": "first-second-inclusion-v1",
                "first_prize_weight": FIRST_PRIZE_BLEND,
                "second_prize_qualifying_weight": SECOND_PRIZE_BLEND,
                "second_combinations_per_draw": MAIN_COUNT,
                "prior_strength_actual_draws": PRIOR_STRENGTH,
                "second_smoothed_formula": "(second_qualifying_hits / 6 + 20 * 6 / 45) / (draw_count + 20)",
                "derived_combinations_are_independent_draws": False,
                "weights_selected_using_holdout_metrics": False,
                "interpretation": "Fixed experimental inclusion-rate mixture; not calibrated winning probabilities or actual winning purchase counts."},
            "overall": _group(clean),
            "by_iso_year": [{"iso_year": year, **_group(group)} for year, group in sorted(years.items())],
            "by_iso_week": [{"iso_week": week, **_group(group)} for week, group in sorted(weeks.items())],
            "by_iso_year_week": compact,
            "target_week": {**_group(target_history),
                            "by_iso_year": [{"iso_year": year, **_group(group)}
                                            for (year, week), group in sorted(year_weeks.items())
                                            if week == iso.week],
                            "first_prize": _repeat_summary(_combination_counts(target_history)),
                            "second_prize_qualifying": _repeat_summary(
                                _combination_counts(target_history, second=True))},
            "combination_repeat_summary": {
                "first_prize": _repeat_summary(_combination_counts(clean)),
                "second_prize_qualifying": _repeat_summary(_combination_counts(clean, second=True))},
            "limitations": ["Historical rates describe the observed samples, not future winning probabilities.",
                            "ISO-week groups are small; tied exact combinations may have no repeated observations.",
                            "TimesFM ranks and blending weights are experimental selection rules, not calibrated probabilities."]}
