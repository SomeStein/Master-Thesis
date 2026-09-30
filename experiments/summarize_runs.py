"""Validate the raw IPA rows and render the thesis comparison table."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CASES = [f"mejstrikpumm{k}" for k in range(251, 258)] + [
    "mejstrik_reif_2x2",
    "controlled_d3_n2_seed2",
    "controlled_d7_n4_seed2001",
    "controlled_d7_n4_seed2033",
]
LABELS = {f"mejstrikpumm{k}": f"MP{k}" for k in range(251, 258)} | {
    "mejstrik_reif_2x2": "MR2",
    "controlled_d3_n2_seed2": "C3",
    "controlled_d7_n4_seed2001": "C7a",
    "controlled_d7_n4_seed2033": "C7b",
}
CODES = {
    "certified": "C",
    "iteration_limit": "I",
    "time_limit": "T",
    "resource_limit": "L",
    "stopped": "S",
    "exception": "E",
    "external_timeout": "X",
    "process_failure": "E",
}


def median(rows: list[dict[str, str]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.median(vals) if vals else math.nan


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else "--"


def outcome(rows: list[dict[str, str]]) -> str:
    return "/".join(sorted({CODES.get(row["status"], "?") for row in rows}))


def table_time(row: dict[str, object]) -> str:
    if row["outcome"] == "X":
        return r"$\geq180$"
    return num(float(row["median_elapsed_s"]), 4)


def main() -> None:
    with (DATA / "ipa_runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["mode"])].append(row)
    for case in CASES:
        for mode in ("off", "strict"):
            actual = {int(row["repeat"]) for row in grouped[(case, mode)]}
            if len(grouped[(case, mode)]) != 3 or actual != {1, 2, 3}:
                raise SystemExit(f"Incomplete data for {case}, {mode}: {sorted(actual)}")
    if len(rows) != 3 * 2 * len(CASES):
        raise SystemExit(f"Unexpected total row count: {len(rows)}")

    summary_rows = []
    for case in CASES:
        for mode in ("off", "strict"):
            group = grouped[(case, mode)]
            summary_rows.append(
                {
                    "case_id": case,
                    "mode": mode,
                    "outcome": outcome(group),
                    "median_elapsed_s": median(group, "elapsed_s"),
                    "median_iterations": median(group, "iterations"),
                    "median_vertices": median(group, "vertices"),
                    "median_jsr_lower": median(group, "jsr_lower"),
                    "median_jsr_upper": median(group, "jsr_upper"),
                    "median_tree_closed_points": median(group, "tree_closed_points"),
                }
            )
    with (DATA / "ipa_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{@{}lcrrcrrr@{}}",
        r"\toprule",
        r"Set & Off & Iter./vert. & Time (s) & Strict & Iter./vert. & Time (s) & Tree points \\",
        r"\midrule",
    ]
    for case in CASES:
        off = next(row for row in summary_rows if row["case_id"] == case and row["mode"] == "off")
        strict = next(row for row in summary_rows if row["case_id"] == case and row["mode"] == "strict")
        lines.append(
            f"{LABELS[case]} & {off['outcome']} & "
            f"{num(off['median_iterations'])}/{num(off['median_vertices'])} & "
            f"{table_time(off)} & {strict['outcome']} & "
            f"{num(strict['median_iterations'])}/{num(strict['median_vertices'])} & "
            f"{table_time(strict)} & "
            f"{num(strict['median_tree_closed_points'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Matched IPA runs on the fixed cohort. Times, counts, and closed tree points are medians of three repeats. A combined outcome code records variation across repeats.}",
        r"\label{tab:ipaComparison}",
        r"\end{table}",
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Set & Off lower & Off upper & Strict lower & Strict upper \\",
        r"\midrule",
    ]
    for case in CASES:
        off = next(row for row in summary_rows if row["case_id"] == case and row["mode"] == "off")
        strict = next(row for row in summary_rows if row["case_id"] == case and row["mode"] == "strict")
        lines.append(
            f"{LABELS[case]} & {num(off['median_jsr_lower'], 12)} & "
            f"{num(off['median_jsr_upper'], 12)} & "
            f"{num(strict['median_jsr_lower'], 12)} & "
            f"{num(strict['median_jsr_upper'], 12)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Median returned JSR lower and upper bounds. Missing values indicate that no bounds were returned. Scalar entries with outcome $C$ in Table~\ref{tab:ipaComparison} are the toolbox's numerical certificates.}",
        r"\label{tab:ipaBounds}",
        r"\end{table}",
    ]
    (DATA / "summary_table.tex").write_text("\n".join(lines) + "\n")
    print(f"Validated {len(rows)} runs; wrote summary CSV and LaTeX table.")


if __name__ == "__main__":
    main()
