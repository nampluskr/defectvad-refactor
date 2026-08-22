import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.io import save_json


@dataclass
class CaseResult:
    case_id: str
    run_name: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    mode: str
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    error_msg: Optional[str] = None
    output_dir: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSummary:
    batch_name: str
    task_name: str
    mode: str
    total_cases: int
    success_cases: int
    failed_cases: int
    skipped_cases: int
    total_elapsed_sec: float
    results: List[CaseResult]
    mean_metrics: Dict[str, float] = field(default_factory=dict)


def compute_mean_metrics(results: List[CaseResult]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in results:
        if r.status == "SUCCESS" and r.metrics:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1

    means = {}
    for k, total in sums.items():
        count = counts[k]
        if count > 0:
            means[k] = round(total / count, 4)
    return means


def save_summary_json(summary: BatchSummary, filepath: str) -> None:
    data = asdict(summary)
    save_json(data, filepath)


def save_summary_csv(summary: BatchSummary, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if not summary.results:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            f.write("No cases found\n")
        return

    # Determine metric headers
    metric_keys = sorted(
        list(
            {
                k
                for r in summary.results
                for k in r.metrics.keys()
                if isinstance(r.metrics.get(k), (int, float))
            }
        )
    )

    meta_keys = ["category", "backbone", "size"]
    used_meta_keys = [
        k
        for k in meta_keys
        if any(k in r.meta or f"data_{k}" in r.meta or f"model_{k}" in r.meta for r in summary.results)
    ]

    fieldnames = ["case_id", "run_name", "status"] + used_meta_keys + metric_keys + ["elapsed_sec", "error_msg"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary.results:
            row = {
                "case_id": r.case_id,
                "run_name": r.run_name,
                "status": r.status,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "error_msg": r.error_msg or "",
            }
            for mk in used_meta_keys:
                row[mk] = r.meta.get(mk, r.meta.get(f"data_{mk}", r.meta.get(f"model_{mk}", "")))
            for k in metric_keys:
                if k in r.metrics:
                    row[k] = round(r.metrics[k], 4)
                else:
                    row[k] = ""
            writer.writerow(row)

        # Write Mean Row if multiple success cases
        if summary.success_cases > 1 and summary.mean_metrics:
            mean_row = {
                "case_id": "MEAN",
                "run_name": f"({summary.success_cases} cases)",
                "status": "SUMMARY",
                "elapsed_sec": round(summary.total_elapsed_sec, 2),
                "error_msg": "",
            }
            for mk in used_meta_keys:
                mean_row[mk] = "-"
            for k in metric_keys:
                if k in summary.mean_metrics:
                    mean_row[k] = summary.mean_metrics[k]
                else:
                    mean_row[k] = ""
            writer.writerow(mean_row)


def render_summary_table(summary: BatchSummary) -> str:
    """Render an ASCII/Markdown-style summary leaderboard."""
    if not summary.results:
        return "No batch results available."

    metric_keys = sorted(
        list(
            {
                k
                for r in summary.results
                for k in r.metrics.keys()
                if isinstance(r.metrics.get(k), (int, float))
            }
        )
    )

    headers = ["Case ID", "Run Name", "Status"] + [k.replace("_", " ").title() for k in metric_keys] + ["Time (s)"]
    rows = []

    for r in summary.results:
        row = [r.case_id, r.run_name, r.status]
        for k in metric_keys:
            if k in r.metrics:
                row.append(f"{r.metrics[k]:.4f}")
            else:
                row.append("-")
        row.append(f"{r.elapsed_sec:.1f}")
        rows.append(row)

    # Add Mean Row
    if summary.success_cases > 1 and summary.mean_metrics:
        mean_row = ["MEAN", f"({summary.success_cases} items)", "SUMMARY"]
        for k in metric_keys:
            if k in summary.mean_metrics:
                mean_row.append(f"{summary.mean_metrics[k]:.4f}")
            else:
                mean_row.append("-")
        mean_row.append(f"{summary.total_elapsed_sec:.1f}")
        rows.append(mean_row)

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    # Build ASCII table
    sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"

    lines = [
        sep_line,
        header_line,
        sep_line,
    ]

    for idx, row in enumerate(rows):
        is_mean = (row[0] == "MEAN")
        if is_mean:
            lines.append(sep_line)
        row_str = "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " |"
        lines.append(row_str)

    lines.append(sep_line)
    lines.append(
        f"Summary: Total={summary.total_cases} | Success={summary.success_cases} | "
        f"Failed={summary.failed_cases} | Skipped={summary.skipped_cases} | "
        f"Elapsed={summary.total_elapsed_sec:.2f}s"
    )

    return "\n".join(lines)
