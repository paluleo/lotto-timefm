"""Offline fixtures only: none of these invented draws is published."""

import copy
import importlib.util
import math
import unittest
from datetime import date, timedelta
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "number_statistics", Path(__file__).resolve().parents[1] / "scripts" / "number_statistics.py")
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def fixture(round_number, numbers=None, bonus=45):
    return {"round": round_number,
            "date": (stats.FIRST_DRAW_DATE + timedelta(weeks=round_number - 1)).isoformat(),
            "numbers": numbers or [1, 2, 3, 4, 5, 6], "bonus": bonus}


class NumberStatisticsTests(unittest.TestCase):
    def test_canonical_validation_and_no_mutation(self):
        original = [fixture(1, [6, 2, 5, 4, 3, 1]), fixture(2)]
        untouched = copy.deepcopy(original)
        result = stats.validate_draws(original)
        self.assertEqual(result[0]["numbers"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(original, untouched)

    def test_invalid_draw_numbers_bonus_dates_and_gaps_rejected(self):
        valid = fixture(1)
        variants = [{**valid, "numbers": [1, 1, 2, 3, 4, 5]},
                    {**valid, "numbers": [True, 2, 3, 4, 5, 6]},
                    {**valid, "numbers": [0, 2, 3, 4, 5, 6]},
                    {**valid, "bonus": 6}, {**valid, "bonus": False},
                    {**valid, "date": "2002-12-08"}, {**valid, "round": True}]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                stats.validate_draws([value])
        for values in [[], [fixture(2), fixture(1)], [fixture(1), fixture(3)], [fixture(1)] * 2]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                stats.validate_draws(values)

    def test_second_prize_derives_exactly_six_qualifying_combinations(self):
        actual = fixture(1)
        combos = stats.second_prize_combinations(actual)
        self.assertEqual(len({tuple(ns) for ns in combos}), 6)
        for ns in combos:
            result = stats.score_game(ns, actual)
            self.assertEqual(result, {"matches": 5, "bonus_match": True, "rank": 2})

    def test_all_rank_conditions(self):
        actual = fixture(1)
        examples = [([1, 2, 3, 4, 5, 6], 1), ([1, 2, 3, 4, 5, 45], 2),
                    ([1, 2, 3, 4, 5, 7], 3), ([1, 2, 3, 4, 7, 8], 4),
                    ([1, 2, 3, 7, 8, 9], 5), ([1, 2, 7, 8, 9, 45], None)]
        for numbers, expected in examples:
            with self.subTest(numbers=numbers):
                self.assertEqual(stats.score_game(numbers, actual)["rank"], expected)

    def test_average_tied_ranks_are_not_number_order_tiebreaks(self):
        ranks = stats.rank_scores([0.0] * 44 + [2.0])
        self.assertEqual(len(set(ranks[:44])), 1)
        self.assertEqual(ranks[0], 21.5 / 44)
        self.assertEqual(ranks[-1], 1)
        self.assertEqual(stats.rank_scores([3.0] * 45), [0.5] * 45)
        self.assertEqual(stats.rank_scores(list(range(-22, 23))), [i / 44 for i in range(45)])
        for invalid in [[0] * 44, [True] + [0] * 44, [math.nan] + [0] * 44, [math.inf] + [0] * 44]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                stats.rank_scores(invalid)

    def test_iso_year_uses_week_year_including_new_year_boundary(self):
        start = (date(2020, 12, 26) - stats.FIRST_DRAW_DATE).days // 7 + 1
        draws = [fixture(start), fixture(start + 1)]
        summary = stats.summarize(draws, "2021-01-09")
        self.assertEqual([(r["iso_year"], r["iso_week"]) for r in summary["by_iso_year_week"]],
                         [(2020, 52), (2020, 53)])
        self.assertEqual(summary["target_iso_year"], 2021)
        self.assertEqual(summary["target_iso_week"], 1)
        self.assertEqual(summary["target_week"]["sample_count"], 0)
        self.assertIsNone(summary["target_week"]["numbers"][0]["main_rate"])
        self.assertAlmostEqual(summary["target_week"]["numbers"][0]["smoothed_main_rate"], 6 / 45)

    def test_summary_preserves_denominators_and_combination_repeats(self):
        draws = [fixture(1), fixture(2), fixture(3, [1, 2, 3, 4, 5, 7])]
        summary = stats.summarize(draws, fixture(4)["date"])
        self.assertEqual(summary["overall"]["sample_count"], 3)
        first = summary["overall"]["numbers"][0]
        self.assertEqual(first["main_hits"], 3)
        self.assertEqual(first["main_rate"], 1)
        self.assertEqual(summary["overall"]["numbers"][44]["bonus_hits"], 3)
        repeats = summary["combination_repeat_summary"]
        self.assertEqual(repeats["first_prize"]["observations"], 3)
        self.assertEqual(repeats["first_prize"]["maximum_count"], 2)
        self.assertEqual(repeats["second_prize_qualifying"]["observations"], 18)
        self.assertIn("not winning ticket purchase records", summary["second_prize_definition"])

    def test_second_inclusion_counts_and_twenty_actual_draw_prior(self):
        summary = stats.summarize([fixture(1)], fixture(2)["date"])
        group = summary["overall"]
        main, bonus, absent = group["numbers"][0], group["numbers"][44], group["numbers"][10]
        self.assertEqual(main["second_qualifying_hits"], 5)
        self.assertEqual(bonus["second_qualifying_hits"], 6)
        self.assertEqual(absent["second_qualifying_hits"], 0)
        self.assertEqual(main["second_qualifying_rate"], 5 / 6)
        self.assertEqual(bonus["second_qualifying_rate"], 1)
        prior = 20 * 6 / 45
        self.assertAlmostEqual(main["smoothed_second_qualifying_rate"], (5 / 6 + prior) / 21)
        self.assertAlmostEqual(bonus["smoothed_second_qualifying_rate"], (1 + prior) / 21)
        self.assertAlmostEqual(main["historical_selection_score"],
                               .75 * main["smoothed_main_rate"] + .25 * main["smoothed_second_qualifying_rate"])
        for key in ["main_rate", "second_qualifying_rate", "smoothed_main_rate",
                    "smoothed_second_qualifying_rate", "historical_selection_score"]:
            self.assertAlmostEqual(sum(row[key] for row in group["numbers"]), 6)
        self.assertFalse(summary["historical_rule"]["derived_combinations_are_independent_draws"])
        self.assertFalse(summary["historical_rule"]["weights_selected_using_holdout_metrics"])

    def test_second_prize_history_changes_ranking_for_bonus_only_numbers(self):
        draws = [fixture(1)]
        target = fixture(2)["date"]
        # 45 and 44 have equal main counts, but only 45 occurred as the bonus.
        old_main_ranks = stats.rank_scores(stats._main_rates(draws))
        historical_ranks = stats.rank_scores(stats._historical_rates(draws))
        self.assertEqual(old_main_ranks[44], old_main_ranks[43])
        self.assertGreater(historical_ranks[44], historical_ranks[43])
        weights = stats.recommendation_weights(draws, target, method="statistics")
        self.assertGreater(weights[44], weights[43])
        altered = [{**draws[0], "bonus": 44}]
        weights_altered = stats.recommendation_weights(altered, target, method="statistics")
        self.assertGreater(weights_altered[43], weights_altered[44])

    def test_weekly_bonus_placement_changes_seasonal_weight_with_same_overall_counts(self):
        history = [fixture(i) for i in range(1, 105)]
        target = fixture(105)["date"]
        target_week = date.fromisoformat(target).isocalendar().week
        in_week = next(i for i, d in enumerate(history)
                       if date.fromisoformat(d["date"]).isocalendar().week == target_week)
        outside_week = next(i for i, d in enumerate(history)
                            if date.fromisoformat(d["date"]).isocalendar().week != target_week)
        a, b = copy.deepcopy(history), copy.deepcopy(history)
        a[in_week]["bonus"] = 44
        b[outside_week]["bonus"] = 44
        self.assertEqual(stats._historical_rates(a), stats._historical_rates(b))
        weights_a = stats.recommendation_weights(a, target, method="statistics")
        weights_b = stats.recommendation_weights(b, target, method="statistics")
        self.assertGreater(weights_a[43], weights_b[43])

    def test_preview_does_not_hide_cutoff_ties(self):
        draws = [fixture(i, list(range(i, i + 6))) for i in range(1, 16)]
        summary = stats.summarize(draws, fixture(16)["date"])
        repeated = summary["combination_repeat_summary"]["first_prize"]
        self.assertEqual(len(repeated["preview"]), 10)
        self.assertEqual(repeated["total_tied_at_cutoff"], 15)
        self.assertEqual(repeated["repeated_combinations"], 0)

    def test_recommendation_rules_are_positive_reproducible_and_causal(self):
        draws = [fixture(i) for i in range(1, 65)]
        target = fixture(65)["date"]
        model = list(range(45))
        for method in ["combined", "statistics", "model", "uniform"]:
            with self.subTest(method=method):
                weights = stats.recommendation_weights(draws, target, model, method)
                self.assertEqual(len(weights), 45)
                self.assertTrue(all(math.isfinite(w) and w > 0 for w in weights))
                games = stats.generate_games(weights, seed=146)
                self.assertEqual(games, stats.generate_games(weights, seed=146))
                self.assertEqual(len({tuple(ns) for ns in games}), 5)
                for ns in games:
                    self.assertEqual(ns, sorted(set(ns)))
                    self.assertEqual(len(ns), 6)
                    self.assertTrue(all(1 <= n <= 45 for n in ns))
        self.assertEqual(stats.recommendation_weights(draws, target, method="uniform"), [1.0] * 45)
        for target in [draws[-1]["date"], draws[0]["date"]]:
            with self.assertRaises(ValueError):
                stats.recommendation_weights(draws, target, model)
        with self.assertRaises(ValueError):
            stats.recommendation_weights(draws, fixture(65)["date"], method="combined")

    def test_extreme_and_invalid_weights_fail_explicitly(self):
        for weights in [[0] * 45, [1] * 44, [math.inf] + [1] * 44,
                        [1e-320] + [1e300] * 44]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                stats.generate_games(weights)
        games = stats.generate_games([1e300] * 45)
        self.assertEqual(len(games), 5)


if __name__ == "__main__":
    unittest.main()
