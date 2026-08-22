import argparse
import glob
import json
import os
import sys
from typing import Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.batch.summary import (
    BatchSummary,
    CaseResult,
    compute_mean_metrics,
    render_summary_table,
    save_summary_csv,
    save_summary_json,
)
from src.utils.io import load_json, load_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CV Experiment Report & Leaderboard Generator",
        allow_abbrev=False,
    )
    parser.add_argument("--dir", "-d", type=str, required=True, help="Path to outputs directory containing experiment runs")
    parser.add_argument("--split", "-s", type=str, default="test", help="Split name to extract metrics for (default: test)")
    parser.add_argument("--csv", type=str, default=None, help="Optional output path to export summary CSV")
    parser.add_argument("--json", type=str, default=None, help="Optional output path to export summary JSON")
    return parser.parse_args()


def scan_directory_for_results(root_dir: str, split: str = "test") -> BatchSummary:
    summary_json_path = os.path.join(root_dir, "summary.json")
    if os.path.isfile(summary_json_path):
        try:
            data = load_json(summary_json_path)
            results = [
                CaseResult(
                    case_id=r["case_id"],
                    run_name=r["run_name"],
                    status=r["status"],
                    mode=r.get("mode", "evaluate"),
                    metrics=r.get("metrics", {}),
                    elapsed_sec=r.get("elapsed_sec", 0.0),
                    error_msg=r.get("error_msg"),
                    output_dir=r.get("output_dir", ""),
                    meta=r.get("meta", {}),
                )
                for r in data.get("results", [])
            ]
            return BatchSummary(
                batch_name=data.get("batch_name", data.get("matrix_name", os.path.basename(root_dir))),
                task_name=data.get("task_name", "cv_task"),
                mode=data.get("mode", "report"),
                total_cases=len(results),
                success_cases=sum(1 for r in results if r.status == "SUCCESS"),
                failed_cases=sum(1 for r in results if r.status == "FAILED"),
                skipped_cases=sum(1 for r in results if r.status == "SKIPPED"),
                total_elapsed_sec=data.get("total_elapsed_sec", sum(r.elapsed_sec for r in results)),
                results=results,
                mean_metrics=compute_mean_metrics(results),
            )
        except Exception:
            pass

    # Fallback: scan subdirectories for metrics_<split>.json
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    results: List[CaseResult] = []

    for d in sorted(subdirs):
        run_name = os.path.basename(d)
        if run_name in ("checkpoints", "logs", "visualizations"):
            continue

        metrics_file = os.path.join(d, f"metrics_{split}.json")
        resolved_config_file = os.path.join(d, "config.resolved.yaml")

        meta = {}
        if os.path.isfile(resolved_config_file):
            try:
                cfg = load_yaml(resolved_config_file)
                meta["category"] = cfg.get("data", {}).get("params", {}).get("category", "")
                meta["backbone"] = cfg.get("model", {}).get("params", {}).get("backbone", "")
                meta["size"] = cfg.get("model", {}).get("params", {}).get("size", "")
            except Exception:
                pass

        if os.path.isfile(metrics_file):
            try:
                metrics = load_json(metrics_file)
                results.append(
                    CaseResult(
                        case_id=meta.get("category") or run_name,
                        run_name=run_name,
                        status="SUCCESS",
                        mode="evaluate",
                        metrics=metrics,
                        elapsed_sec=0.0,
                        output_dir=d,
                        meta=meta,
                    )
                )
            except Exception as e:
                results.append(
                    CaseResult(
                        case_id=meta.get("category") or run_name,
                        run_name=run_name,
                        status="FAILED",
                        mode="evaluate",
                        metrics={},
                        elapsed_sec=0.0,
                        error_msg=f"Invalid metrics JSON: {e}",
                        output_dir=d,
                        meta=meta,
                    )
                )
        else:
            # Run dir without metrics JSON
            results.append(
                CaseResult(
                    case_id=meta.get("category") or run_name,
                    run_name=run_name,
                    status="FAILED",
                    mode="evaluate",
                    metrics={},
                    elapsed_sec=0.0,
                    error_msg=f"metrics_{split}.json not found",
                    output_dir=d,
                    meta=meta,
                )
            )

    success_count = sum(1 for r in results if r.status == "SUCCESS")
    failed_count = sum(1 for r in results if r.status == "FAILED")
    mean_metrics = compute_mean_metrics(results)

    return BatchSummary(
        batch_name=os.path.basename(os.path.normpath(root_dir)),
        task_name="cv_task",
        mode="report",
        total_cases=len(results),
        success_cases=success_count,
        failed_cases=failed_count,
        skipped_cases=0,
        total_elapsed_sec=0.0,
        results=results,
        mean_metrics=mean_metrics,
    )


def main():
    args = parse_args()
    if not os.path.isdir(args.dir):
        print(f"Error: Target directory does not exist: {args.dir}")
        sys.exit(1)

    summary = scan_directory_for_results(args.dir, split=args.split)
    table_str = render_summary_table(summary)
    print("\n" + table_str + "\n")

    if args.csv:
        save_summary_csv(summary, args.csv)
        print(f"Summary CSV saved to: {args.csv}")

    if args.json:
        save_summary_json(summary, args.json)
        print(f"Summary JSON saved to: {args.json}")


if __name__ == "__main__":
    main()
