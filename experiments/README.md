# IPA comparison data

The MATLAB script records ordinary IPA (`hybrid=off`) and the strict hybrid IPA on the same matrix families. It uses the toolbox revision `c67fe5e5c18689450417d364367cbf512f661221` from `origin/hybrid_prework_tommsch`. The unmodified revision cannot parse the supervisor's custom leaf words because `ipa_hybrid_make_leaf_words.m` calls a `tocolumn` helper that is local to another file. Apply the adjacent one-function patch before running.

```sh
git clone /path/to/ttoolboxes /tmp/jsr-ttoolboxes-c67fe5e
git -C /tmp/jsr-ttoolboxes-c67fe5e checkout --detach c67fe5e5c18689450417d364367cbf512f661221
git -C /tmp/jsr-ttoolboxes-c67fe5e apply /path/to/master-thesis/experiments/toolbox-c67fe5e-leaf-words.patch
python3 /path/to/master-thesis/experiments/run_all.py /tmp/jsr-ttoolboxes-c67fe5e
python3 /path/to/master-thesis/experiments/summarize_runs.py
```

From the thesis root, check the saved 4×4 tree data with:

```sh
matlab -nojvm -nosplash -batch "addpath('experiments'); verify_primary_certificate('experiments/data','/tmp/jsr-ttoolboxes-c67fe5e')"
```

Set `IPA_OUTPUT_DIR` to an empty directory when rerunning the cohort without replacing the archived rows. Set `MATLAB_BIN` if MATLAB is installed outside `/opt/homebrew/bin/matlab`. The summarizer validates all 66 rows and regenerates the summary CSV and LaTeX tables from the archived raw file.

The MATLAB script sets Gurobi explicitly, sets `numcore=0` and disables plots, and alternates the order of the two modes across repetitions. The Python driver invokes MATLAB once per case, mode, and repetition, resumes after interruption, and records a separate `external_timeout` outcome if the toolbox does not return within 180 seconds. This watchdog is needed because IPA checks its own 120-second limit only between operations. Returned-call elapsed times exclude MATLAB startup; external-timeout elapsed times are censored process durations and include startup. Each row of `data/ipa_runs.csv` is one IPA invocation. `certified` denotes a scalar JSR result with an exact-result exit code; a limit row says only that the run did not finish within its configured limit. `data/primary_certificate.mat` contains the returned tree words and vertex data for the 4×4 example. `verify_primary_certificate.m` checks the stored words, closed tree points, generator spectra, and spanning vertices. `data/run_manifest.txt` records the software and common options; `data/machine.txt` records the hardware. The raw MATLAB MAT file is retained because the custom tree words must be checked against the diagram in the thesis.

The custom tree's admissibility is a mathematical condition in addition to the toolbox's leaf-norm checks. For the main example, generator `[2]` has uncovered word `[-1;1]` and a covered `2` branch. Generator `[2;2]` has uncovered words `[-2;1;1;1]`, `[-2;1;1;2]`, `[-2;1;2;1]`, `[-2;1;2;2]`, `[-2;2;1]` and a covered `2;2` branch. These words exhaust the positive children of every non-leaf node.

`data/pilot_251_same_process.csv` archives six exploratory MP251 rows collected in one MATLAB process. They are excluded from `ipa_runs.csv` and from the thesis tables because their timing setup differs from the final one-process-per-invocation protocol. The main CSV contains a complete uniform rerun of MP251.
