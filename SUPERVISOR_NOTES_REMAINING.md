# Remaining Supervisor Notes (Current Build)

This file lists only the items I did **not** implement because they require your decision, missing bibliography entries, or additional mathematical detail.

Reference PDF: `build/masterthesis.pdf` (42 pages, build dated 2026-03-11).

## 1) Introduction

1. **Citations for subdivision schemes and capacities of codes**
Page(s): **7-8** (Section 1.2)
Location: `chapters/introduction.tex`, motivation paragraph after the bounded/stability bullets.
Needed change: Replace/extend citations to match your supervisor’s requested sources (CDM/Daubechies for subdivision schemes; Moision-Orlitsky-Siegel 2001 for code capacities).
Why pending: These exact bibliography entries are currently not present in `literature.bib`.

2. **Equation layout / spacing refinements in theoretical background**
Page(s): **9-12** (Section 1.3)
Location: `chapters/introduction.tex`, long displayed derivations around the three-member inequality proof.
Needed change: Manual typography cleanup for display density/line breaks according to supervisor preference.
Why pending: This is mainly stylistic and requires your preferred final layout.

## 2) Invariant Polytope Chapter

1. **Add a concrete example (optional but requested)**
Page(s): **17-18** (Section 2.1)
Location: `chapters/invariant_polytope.tex`, after Theorem 2.1.2 / around definitions of invariant and extremal norms.
Needed change: Add a short matrix-family example that illustrates a Barabanov norm and an extremal norm.
Why pending: No canonical example was specified in the notes.

2. **Promote termination conditions to theorem-level statement**
Page(s): **20** (Section 2.4)
Location: `chapters/invariant_polytope.tex`, section `Termination Conditions`.
Needed change: Replace short prose with a theorem (or theorem + remark) stating clear termination assumptions.
Why pending: Requires choosing an exact formal statement and citations you want to present.

## 3) Finite-Tree Chapter

1. **Decide whether to keep Section 3.3 as standalone**
Page(s): **26** (Section 3.3)
Location: `chapters/finite_tree.tex`, section `Termination Conditions`.
Needed change: Either merge this material into Section 3.2 or expand it so a dedicated section is justified.
Why pending: This is a structural choice.

## 4) Hybrid Chapter

1. **Further explanation depth in proof-strategy area**
Page(s): **28-30** (Sections 4.4.1 and 4.4.2)
Location: `chapters/hybrid_approach.tex`, proof-strategy narrative and figure context.
Needed change: Add the level of detail your supervisor requested (decomposition intuition, recursive bounding narrative, etc.).
Why pending: Requires your preferred proof exposition depth and style.

2. **Expand implementation details**
Page(s): **31** (Section 4.6)
Location: `chapters/hybrid_approach.tex`, `Implementation Considerations`.
Needed change: Add concrete implementation choices (tree selection policy, fallback behavior, complexity/memory notes).
Why pending: Requires your actual implementation decisions and experimental constraints.

