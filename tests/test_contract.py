"""Cross-language contract regression using test-only reports held in memory.

No fixture or output is written to public/data. Exact zone-frequency ties used
to sort differently in Python and JavaScript because float summation differs.
"""

import importlib.util
import json
import random
import shutil
import subprocess
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contract_pipeline", ROOT / "scripts" / "fetch_and_predict.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
NOW = datetime(2026, 11, 8, 10, tzinfo=timezone.utc)


def test_only_report(count, rng):
    """Synthetic data for this process only; never fetched or published."""
    latest = pipeline.latest_expected_round(NOW)
    draws = []
    for number in range(latest - count + 1, latest + 1):
        values = rng.sample(range(1, 46), 7)
        draws.append({"round": number, "date": pipeline.round_draw_date(number).isoformat(),
                      "numbers": sorted(values[:6]), "bonus": values[6]})
    return pipeline.build_report(draws, NOW)


class ProducerBrowserContractTests(unittest.TestCase):
    def test_browser_accepts_python_reports_for_every_supported_count(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node.js is required for the producer/browser contract test")
        rng = random.Random(87531)
        cases = []
        # Preserve the sequence that originally reproduced failures at counts
        # 21, 64 and 511, including the default production count of 64.
        for count in (20, 21, 63, 64, 511, 512):
            for trial in range(10):
                cases.append({"label": f"original-reproduction/{count}/{trial}",
                              "report": test_only_report(count, rng)})
        for count in range(pipeline.MIN_DRAWS, pipeline.MAX_DRAWS + 1):
            cases.append({"label": f"supported-count/{count}",
                          "report": test_only_report(count, rng)})

        javascript = r"""
const app = require(process.argv[1]);
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', part => { input += part; });
process.stdin.on('end', () => {
  const cases = JSON.parse(input);
  const failures = [];
  for (const {label, report} of cases) {
    try {
      app.validateForecast(report, Date.parse('2026-11-08T11:00:00Z'));
      if (app.effectiveStatus(report, Date.parse('2026-11-14T11:59:59Z')) !== 'current')
        throw new Error('Report expired before the KST draw cutoff');
      if (app.effectiveStatus(report, Date.parse('2026-11-14T12:00:00Z')) !== 'stale')
        throw new Error('Report remained current after the KST draw cutoff');
    } catch (error) { failures.push({label, error: error.message}); }
  }
  process.stdout.write(JSON.stringify({checked: cases.length, failures}));
  if (failures.length) process.exitCode = 1;
});
"""
        result = subprocess.run(
            [node, "-e", javascript, str(ROOT / "public" / "app.js")],
            input=json.dumps(cases, ensure_ascii=True), text=True, encoding="utf-8",
            capture_output=True, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["checked"], 553)
        self.assertEqual(summary["failures"], [])


if __name__ == "__main__":
    unittest.main()
