import copy
import unittest

from scripts.publish_timesfm_report import public_report


class PublishReportTests(unittest.TestCase):
    def test_projection_preserves_predictions_and_validation(self):
        original = {
            "schema_version": 1, "status": "success", "model": {"timesfm_used": True},
            "prediction": {"raw_scores": {"1": -0.03}},
            "recommendations": {"methods": {"combined": {"games": [[1, 2, 3, 4, 5, 6]]}}},
            "validation": {"rows": [{"target_round": 1249}], "metrics": {"combined": 0.8}},
            "statistics": {"overall": {"sample_count": 1249}, "by_iso_year": [1],
                           "by_iso_week": [2], "by_iso_year_week": [3],
                           "target_week": {"sample_count": 23, "by_iso_year": [4]}}
        }
        snapshot = copy.deepcopy(original)
        published = public_report(original)
        self.assertEqual(original, snapshot)
        for key in ("prediction", "recommendations", "validation"):
            self.assertEqual(published[key], original[key])
        self.assertEqual(published["statistics"]["target_week"], {"sample_count": 23})
        self.assertEqual(published["statistics"]["detail_source"], "./data/draw-history.json")
        self.assertNotIn("by_iso_year_week", published["statistics"])

    def test_failed_status_has_no_fabricated_results(self):
        report = {"schema_version": 1, "status": "failed", "prediction": None}
        self.assertEqual(public_report(report), report)

    def test_cannot_publish_mock_success_as_actual_model(self):
        with self.assertRaises(ValueError):
            public_report({"schema_version": 1, "status": "success", "model": {"timesfm_used": False}})


if __name__ == "__main__":
    unittest.main()
