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

## Working Conventions for Agents
- Default build action for thesis changes: run the configured `latexmk` recipe.
- Preserve existing mathematical notation macros and theorem environment structure in `masterthesis.tex`.
- Keep terminology consistent across edits: JSR, invariant polytope algorithm, finite-tree algorithm, hybrid tree-polytope approach.
- Keep edits minimal and thesis-focused unless explicitly asked for refactors.

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
