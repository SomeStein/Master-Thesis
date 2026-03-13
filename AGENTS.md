# Workspace Codex Context

## Project Identity
- Repository purpose: Produce a mathematics master's thesis on Joint Spectral Radius (JSR) computation.
- Thesis title: "Termination results for hybrid approach of Joint Spectral Radius computation".
- Primary contributor: Aaron Pumm (single contributor).
- Supervisor: Assoz. Prof. Dr. Martin Ehler.
- Co-supervisor: Dr. Thomas Mejstrik MMag.
- Algorithm context: MATLAB is used for algorithm development, based on a previously created JSR algorithm framework by Thomas Mejstrik.

## LaTeX Build Process (Source of Truth: `.vscode/settings.json`)
- Main document: `masterthesis.tex`.
- Output directory: `build/`.
- Tool/recipe name: `latexmk`.
- Configured command and arguments:
  - `latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=%OUTDIR% %DOC%`
- Equivalent direct CLI command from repository root:
  - `latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=build masterthesis.tex`
- LaTeX Workshop clean mode:
  - `latex-workshop.latex.clean.subfolder.enabled = true`
  - `latex-workshop.latex.autoClean.run = onFailed`
  - configured clean file types:
    - `*.aux`, `*.bbl`, `*.blg`, `*.idx`, `*.ind`, `*.lof`, `*.lot`, `*.out`, `*.toc`
    - `*.acn`, `*.acr`, `*.alg`, `*.glg`, `*.glo`, `*.gls`, `*.ist`
    - `*.fls`, `*.log`, `*.fdb_latexmk`, `*.synctex.gz`
- Git behavior for build artifacts:
  - only `build/masterthesis.pdf` should stay tracked; other generated files are ignored.

## Python Environment
- Python-related files are in `src/`.
- Dependency lock file: `src/requirements.txt` (pinned versions).
- Script imports currently used:
  - `numpy`, `plotly`, `matplotlib`, `sympy` (plus standard dependency chain from requirements).
- Local interpreter status at config creation:
  - `python` resolves to `C:\Users\Aaron\AppData\Local\Microsoft\WindowsApps\python.exe`.
  - direct `python` execution currently fails in this environment.
  - `py` launcher is not available.
- If Python work is requested, create and use a local venv first:
  1. `python -m venv .venv`
  2. `.\.venv\Scripts\Activate.ps1`
  3. `pip install -r src/requirements.txt`

## Thesis Source Layout
- Root LaTeX file: `masterthesis.tex`.
- Included chapter files (in order):
  1. `chapters/abstract.tex`
  2. `chapters/introduction.tex`
  3. `chapters/invariant_polytope.tex`
  4. `chapters/finite_tree.tex`
  5. `chapters/hybrid_approach.tex`
  6. `chapters/numerical_testing.tex`
  7. `chapters/conclusion.tex`
- Bibliography:
  - data file: `literature.bib`
  - citation style: `natbib` with `apalike`.

## Local Reference Paths
- General Obsidian project note for this thesis: `/Users/aaronpumm/Library/Mobile Documents/iCloud~md~obsidian/Documents/SteinVault/University/Master/JSR - Project.md`
- Repository root on disk: `/Users/aaronpumm/Desktop/Projekte Lokal/GitHub/Master-Thesis`
- Workspace file inventory at the time of this note:
  - Root and config files:
    - `.gitignore`
    - `.vscode/settings.json`
    - `AGENTS.md`
    - `LICENSE`
    - `README.md`
    - `literature.bib`
    - `masterthesis.tex`
  - Chapter sources:
    - `chapters/abstract.tex`
    - `chapters/appendix_notation.tex`
    - `chapters/conclusion.tex`
    - `chapters/danksagung.tex`
    - `chapters/finite_tree.tex`
    - `chapters/hybrid_approach.tex`
    - `chapters/introduction.tex`
    - `chapters/invariant_polytope.tex`
    - `chapters/numerical_testing.tex`
    - `chapters/titlepage.tex`
  - Python and helper files:
    - `src/requirements.txt`
    - `src/test_JNF.py`
    - `src/testing_polytope_def.py`
    - `src/visualize_bounded_product_trajectories.py`
  - Image assets:
    - `images/newplot.png`
    - `images/univie.jpg`
  - Build output files:
    - `build/masterthesis.pdf`
    - `build/masterthesis.aux`
    - `build/masterthesis.bbl`
    - `build/masterthesis.blg`
    - `build/masterthesis.fdb_latexmk`
    - `build/masterthesis.fls`
    - `build/masterthesis.log`
    - `build/masterthesis.out`
    - `build/masterthesis.synctex.gz`
    - `build/masterthesis.toc`
    - `build/chapters/abstract.aux`
    - `build/chapters/appendix_notation.aux`
    - `build/chapters/conclusion.aux`
    - `build/chapters/finite_tree.aux`
    - `build/chapters/hybrid_approach.aux`
    - `build/chapters/introduction.aux`
    - `build/chapters/invariant_polytope.aux`
    - `build/chapters/numerical_testing.aux`
    - `build/chapters/titlepage.aux`
- Zotero storage root for local paper PDFs: `/Users/aaronpumm/Zotero/storage`
- Zotero folders are 8-character storage keys. Only JSR-related papers are listed below; non-JSR Zotero material is intentionally omitted from this file.
- The following are the priority papers for thesis work and should usually be checked first:
  - Main paper underlying this thesis and the hybrid chapter: `YP8PBE6S/Mejstrik und Reif - 2025 - A hybrid approach to joint spectral radius computation.pdf`
  - Main invariant-polytope reference: `NGJFW3WF/Guglielmi und Protasov - 2013 - Exact Computation of Joint Spectral Characteristics of Linear Operators.pdf`
  - Main finite-tree reference: `H9B8IJGI/Möller und Reif - 2014 - A tree-based approach to joint spectral radius determination.pdf`
  - Main general JSR background reference: `2RERDD9B/Jungers - The Joint Spectral Radius.pdf`
  - Mejstrik invariant-polytope improvement references: `ZCEZZF5F/Mejstrik - 2018 - Improved invariant polytope algorithm and applications.pdf` and `T9ZZU4WI/Mejstrik - 2020 - Algorithm 1011.pdf`
- JSR, invariant polytope, finite-tree, Barabanov, finiteness, and hybrid references:
  - `2H9J623A/Chang und Blondel - 2011 - Approximating the joint spectral radius using a genetic algorithm framework.pdf`
  - `2HJ9WZQ8/Dai - 2013 - Some criteria for spectral finiteness of a finite subset of the real matrix space Rd×d.pdf`
  - `2M96STDY/Guglielmi und Protasov - 2015 - Invariant polytopes of linear operators with applications to regularity of wavelets and of subdivisi.pdf`
  - `2P7G7ZPV/Chang und Blondel - 2013 - An experimental study of approximation algorithms for the joint spectral radius.pdf`
  - `2RERDD9B/Jungers - The Joint Spectral Radius.pdf`
  - `2S5YTSRX/Bochi und Laskawiec - 2023 - Spectrum maximizing products are not generically unique.pdf`
  - `57E9TW4X/Jungers et al. - 2014 - Lifted polytope methods for computing the joint spectral radius.pdf`
  - `58VBD5SI/Guglielmi und Protasov - 2023 - Computing the spectral gap of a family of matrices.pdf`
  - `5F8HJGYU/Kozyakin - 2008 - A relaxation scheme for computation of the joint spectral radius of matrix sets.pdf`
  - `74UHMQ3W/Guglielmi und Zennaro - 2015 - Canonical construction of polytope barabanov norms and antinorms for sets of matrices.pdf`
  - `B7T25DIC/Dai und Kozyakin - 2011 - Finiteness property of a bounded set of matrices with uniformly sub-peripheral spectrum.pdf`
  - `BY3IY9GG/2015 - A note on the joint spectral radius.pdf`
  - `EQAFPZMY/Wang und Wen - 2013 - The finiteness conjecture for the joint spectral radius of a pair of matrices.pdf`
  - `FCJEDCRF/Liu und Xiao - 2011 - Rank-one approximation of joint spectral radius of finite matrix family ∗.pdf`
  - `FEYKATFC/Morris - 2011 - Mather sets for sequences of matrices and applications to the study of joint spectral radii.pdf`
  - `GVBTBYAB/Dai - 2011 - The finite-step realizability of the joint spectral radius of a pair of d×d matrices one of which be.pdf`
  - `H9B8IJGI/Möller und Reif - 2014 - A tree-based approach to joint spectral radius determination.pdf`
  - `K57HABQA/Dai - 2011 - A criterion of simultaneously symmetrization and spectral finiteness for a finite set of real 2-by-2.pdf`
  - `NGJFW3WF/Guglielmi und Protasov - 2013 - Exact Computation of Joint Spectral Characteristics of Linear Operators.pdf`
  - `P2CA8Y95/Kozyakin - 2024 - On pairs of spectrum maximizing products with distinct factor multiplicities.pdf`
  - `P2GJ5L9U/Möller - 2015 - A new strategy for exact determination of the joint spectral radius.pdf`
  - `P4FAHSTA/Morris - 2015 - Generic properties of the lower spectral radius for some low-rank pairs of matrices.pdf`
  - `Q2Z5NY76/Mejstrik - 2025 - The finiteness conjecture for 3×3 binary matrices.pdf`
  - `QFNENRLL/Protasov - 2021 - The barabanov norm is generically unique, simple, and easily computed.pdf`
  - `RGKN79JT/Morris - 2011 - Rank one matrices do not contribute to the failure of the finiteness property.pdf`
  - `T9ZZU4WI/Mejstrik - 2020 - Algorithm 1011.pdf`
  - `UJ3WF8RG/Guglielmi und Zennaro - 2009 - Finding extremal complex polytope norms for families of real matrices.pdf`
  - `US6LST87/Bochi und Morris - 2013 - Continuity properties of the lower spectral radius.pdf`
  - `UZTAR85G/Liu und Xiao - 2012 - Computation of joint spectral radius for network model associated with rank-one matrix set.pdf`
  - `VN68LZWS/Vankeerberghen et al. - 2014 - JSR a toolbox to compute the joint spectral radius.pdf`
  - `WB2JUKL5/Morris - 2009 - Criteria for the stability of the finiteness property and for the uniqueness of Barabanov norms.pdf`
  - `Y4BI4LNZ/Cicone et al. - 2010 - Finiteness property of pairs of 2× 2 sign-matrices via real extremal polytope norms.pdf`
  - `YP8PBE6S/Mejstrik und Reif - 2025 - A hybrid approach to joint spectral radius computation.pdf`
  - `ZCEZZF5F/Mejstrik - 2018 - Improved invariant polytope algorithm and applications.pdf`
  - `ZPV24TXZ/Guglielmi und Zennaro - 2014 - Stability of linear problems Joint spectral radius of sets of matrices.pdf`

## Working Conventions for Agents
- Default build action for thesis changes: run the configured `latexmk` recipe.
- Preserve existing mathematical notation macros and theorem environment structure in `masterthesis.tex`.
- Keep terminology consistent across edits: JSR, invariant polytope algorithm, finite-tree algorithm, hybrid tree-polytope approach.
- Keep edits minimal and thesis-focused unless explicitly asked for refactors.

## Editorial Constraints from Thomas Notes
- Treat the following as thesis-wide defaults unless a chapter-specific requirement clearly overrides them.
- Keep prose direct and precise; avoid vague fillers such as "some", "very large", or "methodology" when a more exact formulation is available.
- Keep terminology fixed; do not alternate between near-synonyms for the same concept unless the distinction is intentional and explicitly defined.
- Keep sentences short. As a default target, no sentence should run longer than about three manuscript lines.
- Avoid repeating the thesis motivation or the same high-level claim multiple times unless repetition is structurally necessary.
- When introducing notation, write complete grammatical sentences instead of leaving a standalone "Let ..." fragment.
- Define every symbol, set cardinality, index range, and standing assumption at first use.
- Avoid dangling symbols in prose; if a variable such as `\lambda` appears in a sentence, state clearly what it denotes.
- Avoid symbolic quantifiers such as `\forall` and `\exists` in prose, displayed equations, and pseudocode; write them out in words.
- Do not introduce terminology that is unnecessary for the argument, and do not rename an existing concept later without explanation.
- Give the reader brief context around important theorems, definitions, and algorithms; add at least a short line explaining why the item is needed when the role is not obvious.
- Explain a displayed computation before it appears, or in the same sentence; do not place the explanation only afterwards.
- Keep displayed formulas compact and readable; avoid unnecessarily tall multiline layouts or excessive whitespace.
- Facts not needed for a proof should usually be moved to a remark instead of interrupting the proof flow.
- Cite the earliest or standard source for classical results whenever possible. If the proof is reproduced from a different source, say so explicitly.
- Pseudocode should name inputs and important variables explicitly and avoid unexplained notation.
- Use thesis notation and typography consistently, including existing macros such as `\JSR` and standard commands such as `\ldots`.
- When editing a passage, also fix obvious nearby typos, grammar issues, and capitalization inconsistencies.

## Frequently Used Notation
- `\mathcal{A} = \{A_1,\dots,A_n\}`: base matrix family, with `n = |\mathcal{A}|` and matrix dimension `d`.
- `\tilde{\mathcal{A}}`: scaled matrix family after preprocessing.
- `\mathcal{G} = \{G_1,\dots,G_m\}`: generator set, with `m = |\mathcal{G}|`.
- `\mathcal{I} = \{1,\dots,n\}`, `I \in \mathcal{I}^k`, `A_I = A_{i_k}\cdots A_{i_1}`: positive-index products.
- `\mathcal{J} = \{-m,\dots,-1,1,\dots,n\}`, `J \in \mathcal{J}^k`, `A_J`: extended index encoding for tree/generator products.
- `\JSR(\mathcal{A})`: joint spectral radius operator.
- `\|\cdot\|_{\mathrm{co}_{\mathrm{s}}(V)} = \|\cdot\|_V`: polytope norm notation.
- `\mathcal{L}(T)`: leafage of a tree `T`.
- `(\mathcal{A},\mathcal{G})`-tree and `T_{\min(\mathcal{A},\mathcal{G})}`: tree notation used in finite-tree/hybrid chapters.
