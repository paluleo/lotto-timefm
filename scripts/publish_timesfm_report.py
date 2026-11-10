"""Publish a compact view of the actual model report without changing results."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

if __package__:
    from .timesfm_forecast import ModelExecutionError, atomic_write, failed_report
else:
    from timesfm_forecast import ModelExecutionError, atomic_write, failed_report


def public_report(report: dict) -> dict:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("unsupported TimesFM report")
    if report.get("status") not in {"success", "failed"}:
        raise ValueError("invalid TimesFM report status")
    result = copy.deepcopy(report)
    if result["status"] == "success":
        if result.get("model", {}).get("timesfm_used") is not True:
            raise ValueError("successful report must contain actual model inference")
        statistics = result.get("statistics")
        if not isinstance(statistics, dict) or not isinstance(statistics.get("target_week"), dict):
            raise ValueError("missing verified number statistics")
        for name in ("by_iso_year", "by_iso_week", "by_iso_year_week"):
            statistics.pop(name, None)
        statistics["target_week"].pop("by_iso_year", None)
        statistics["detail_source"] = "./data/draw-history.json"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--fail-reason", help="Publish an explicit incomplete-run status, never replacement scores")
    parser.add_argument("--output", type=Path, default=Path("public/data/timesfm.json"))
    args = parser.parse_args()
    if args.input and args.input.resolve() == args.output.resolve():
        parser.error("preserve the complete input audit artifact")
    if args.fail_reason:
        report = failed_report(ModelExecutionError("pipeline", args.fail_reason), context=64)
    else:
        report = json.loads(args.input.read_text(encoding="utf-8-sig"))
    atomic_write(args.output, public_report(report))
    print(f"Published TimesFM status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
