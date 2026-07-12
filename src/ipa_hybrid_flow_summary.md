# IPA Hybrid Flow Summary

## Workspace Roles

- `master-thesis` contains the LaTeX thesis, Python helper scripts, and thesis-facing notes. The static summary in this file is meant as a bridge from implementation details to the numerical chapter.
- `ttoolboxes` contains the MATLAB implementation of `ipa`, including the hybrid tree-polytope code in `ttoolboxes/ipa/private/ipa_hybrid_*`.
- `ttest` contains the MATLAB test framework used by the toolbox tests, including `unittest/test_ipa_hybrid.m`.

## Current IPA Flow

The public entry point is `ipa`. It first parses matrix data and options through `ipa_option`, then performs basic checks and invariant-subspace splitting. For each block it builds a block variable, resets the per-run state, and calls `makecandidate`.

`makecandidate` selects or accepts spectrum-maximizing product candidates, computes the scaling value `lambda`, obtains leading eigenvectors, removes unsuitable duplicates or interior starting vectors, scales the matrices, and adds explicit or automatic extra vertices. The result is the initial cyclic-tree data used by the worker.

After candidate construction, `ipa` balances the starting vectors, constructs the cyclic tree with `ipa_cyclictree`, applies optional delta scaling, recomputes roots if necessary, and calls `ipa_hybrid_prepare` when hybrid mode is enabled.

The worker loop then repeats the same high-level IPA cycle: generate children with `ipa_generatenewvertex`, optionally run the hybrid filter, estimate norms, select vertices for exact norm computation, compute exact polytope norms, save the results back into the cyclic tree, and test termination. `ipa_combinevar` merges block logs and returns the final `JSR` interval and `nfo`.

## Hybrid Preparation

Hybrid mode is configured by `hybrid_mode`, with accepted values `off`, `strict`, and `performance`. The legacy `hybrid` flag enables `strict` mode.

`ipa_hybrid_prepare` chooses one generator, normally the first candidate ordering, and constructs the default minimal signed leaf tree via `ipa_hybrid_make_minimal_tree`. A negative first index represents an arbitrary non-negative power of the generator. For a generator `G` and tail `X`, a leaf `[-1; tail]` represents all products `X G^n`.

For infinite leaves, `ipa_hybrid_leading_data` computes eigenbasis data for the generator. The hybrid code needs a certifiable periodic leading part and a strict stable part. It also prepares cached tail products and limit maps. Runtime safeguards are tracked through cumulative, per-check, and adaptive budgets.

## Strict Mode

In `strict` mode, `ipa_hybrid_filter_vertices` runs immediately after new vertices are generated and before ordinary norm estimation. It tests newly generated, still-uncomputed vertices against the current polytope.

For finite leaves, it checks whether the leaf image lies strictly inside the current symmetric polytope. For infinite leaves, `ipa_hybrid_check_infinitypath` checks periodic limit points, computes a tail threshold from the stable spectral radius, and then checks the finite initial powers up to that threshold. If all leaves close, the vertex norm is written as inside and the ordinary exact norm computation is skipped.

## Performance Mode

In `performance` mode, the hybrid filter runs after the usual vertex selection step. It receives the exact-norm selection mask and tests only vertices that would otherwise be sent to exact polytope norm computation.

The implementation batches points whenever possible. If hybrid closure changes the selected set, IPA reselects vertices before exact norm computation. A return-on-investment gate disables performance mode when many checks remain unresolved and too few close.

## Profiling Acceptance Criterion

The discovery script accepts an example only when the optimized hybrid path is actually used. In practice this means both `strict` and `performance` runs must have positive hybrid closures, positive limit checks, positive tail checks, positive finite-power work, no budget exhaustion, no ROI disable, and fewer exact polytope-norm points than the baseline run.

This criterion excludes runs where hybrid mode was merely enabled but fell back to ordinary IPA behavior.
