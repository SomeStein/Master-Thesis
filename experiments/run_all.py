"""Run each IPA invocation in its own MATLAB process with a hard watchdog.

The toolbox checks maxtime only between internal operations. A slow operation can
therefore exceed 120 seconds; the 180-second process watchdog records that
outcome and continues the predetermined cohort. Existing CSV keys are skipped,
which makes an interrupted experiment resumable.
"""

from __future__ import annotations

import csv
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("IPA_OUTPUT_DIR", str(ROOT / "data")))
CSV = DATA / "ipa_runs.csv"
LOG = DATA / "run_log.txt"
TOOLBOX = Path(sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/jsr-ttoolboxes-c67fe5e")
MATLAB = os.environ.get("MATLAB_BIN", "/opt/homebrew/bin/matlab")
HARD_SECONDS = 180
CASES = ["mejstrikpumm257", "mejstrik_reif_2x2"] + [
    f"mejstrikpumm{k}" for k in range(251, 257)
] + [
    "controlled_d3_n2_seed2", "controlled_d7_n4_seed2001", "controlled_d7_n4_seed2033"
]
DIMS = {"mejstrikpumm251": 2, "mejstrikpumm252": 3,
        "mejstrikpumm253": 3, "mejstrikpumm254": 2,
        "mejstrikpumm255": 2, "mejstrikpumm256": 2,
        "mejstrikpumm257": 4, "mejstrik_reif_2x2": 2,
        "controlled_d3_n2_seed2": 3,
        "controlled_d7_n4_seed2001": 7, "controlled_d7_n4_seed2033": 7}
FIELDS = ["case_id", "category", "dimension", "num_matrices", "repeat", "mode",
          "elapsed_s", "jsr_lower", "jsr_upper", "errorcode", "status", "iterations",
          "vertices", "tree_closed_points", "error_identifier"]


def existing_keys() -> set[tuple[str, str, str]]:
    if not CSV.exists():
        return set()
    with CSV.open(newline="") as stream:
        return {(r["case_id"], r["repeat"], r["mode"]) for r in csv.DictReader(stream)}


def record_failure(case: str, rep: int, mode: str, elapsed: float,
                   status: str, error: str) -> None:
    new = not CSV.exists()
    with CSV.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(dict(case_id=case,
                             category="gallery" if case.startswith("mejstrikpumm") else
                             "published" if case == "mejstrik_reif_2x2" else "controlled",
                             dimension=DIMS[case],
                             num_matrices=4 if case.startswith("controlled_d7") else 2,
                             repeat=rep, mode=mode, elapsed_s=f"{elapsed:.8f}",
                             jsr_lower="nan", jsr_upper="nan", errorcode="nan",
                             status=status, iterations="nan", vertices="nan",
                             tree_closed_points="nan", error_identifier=error))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    keys = existing_keys()
    for case, rep in itertools.product(CASES, range(1, 4)):
        modes = ("off", "strict") if rep % 2 else ("strict", "off")
        for mode in modes:
            key = (case, str(rep), mode)
            if key in keys:
                continue
            expr = (f"addpath('{ROOT}'); "
                    f"run_ipa_comparison('{TOOLBOX}','{DATA}',3,'{case}',{rep},'{mode}')")
            started = time.monotonic()
            print(f"Running {case} {rep} {mode}", flush=True)
            try:
                result = subprocess.run([MATLAB, "-nojvm", "-nosplash",
                                         "-batch", expr], capture_output=True, text=True,
                                        timeout=HARD_SECONDS)
                elapsed = time.monotonic() - started
                with LOG.open("a") as log:
                    log.write(f"\n### {case} {rep} {mode}: exit {result.returncode}, {elapsed:.2f}s\n")
                    log.write(result.stdout[-8000:] + result.stderr[-4000:])
                if key not in existing_keys():
                    record_failure(case, rep, mode, elapsed, "process_failure",
                                   f"matlab_exit_{result.returncode}")
            except subprocess.TimeoutExpired as exc:
                elapsed = time.monotonic() - started
                with LOG.open("a") as log:
                    log.write(f"\n### {case} {rep} {mode}: hard timeout, {elapsed:.2f}s\n")
                    for output in (exc.stdout, exc.stderr):
                        if output:
                            log.write(output.decode(errors="replace")[-4000:]
                                      if isinstance(output, bytes) else output[-4000:])
                if key not in existing_keys():
                    record_failure(case, rep, mode, elapsed, "external_timeout",
                                   "process_watchdog_180s")
            keys.add(key)
            print(f"Completed {case} {rep} {mode}: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
