# Master Thesis

Termination results for a hybrid approach to Joint Spectral Radius (JSR) computation.

## Overview

This repository contains the LaTeX source for a master's thesis focused on:

- theoretical background of the joint spectral radius,
- invariant-polytope and finite-tree algorithms,
- a hybrid algorithmic approach,
- termination properties and numerical experiments.

The main document is compiled from `masterthesis.tex` and chapter files in `chapters/`.

## Repository Structure

- `masterthesis.tex`: root LaTeX document
- `chapters/`: thesis chapters and front matter components
- `literature.bib`: BibTeX database
- `images/`: static image assets
- `.vscode/settings.json`: LaTeX Workshop build configuration
- `build/`: generated build artifacts (only `build/masterthesis.pdf` is tracked)

## Build Workflow (VS Code)

This project is configured for the **LaTeX Workshop** extension in VS Code.

- Output directory: `build/`
- Tool/recipe: `latexmk`
- Main file: `masterthesis.tex`

### Recommended steps

1. Open the repository in VS Code.
2. Ensure the LaTeX Workshop extension is installed.
3. Open `masterthesis.tex`.
4. Run the LaTeX Workshop build recipe (`latexmk`).

The final PDF is written to:

- `build/masterthesis.pdf`

## Bibliography

- Bibliography backend: BibTeX
- Database file: `literature.bib`
- Style in `masterthesis.tex`: `apalike`

## Cleaning Build Artifacts

Use LaTeX Workshop clean actions (configured in `.vscode/settings.json`) to remove intermediate files.

## License

This project is licensed under the MIT License. See `LICENSE`.
