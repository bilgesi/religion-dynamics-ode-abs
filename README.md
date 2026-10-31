# Religion dynamics: reconciled manuscript and reproducibility package

This is the executed revision of **An Epidemiological Modeling Take on Religion Dynamics**.
It contains the revised manuscript, the supplied bibliography with targeted corrections and new data references, all nine original-named PDF figures, three additional holdout figures, and the corresponding code, data and numerical results.

## Read or compile the paper

Open `paper/main.pdf`. The upload-ready LaTeX project is the entire `paper/` directory. Compile `main.tex` with PDFLaTeX, BibTeX, and PDFLaTeX twice. `main.tex` is self-contained as to numerical tables and macros; it still requires `biblio.bib` and the accompanying PDF figures. `manuscript_template.tex` is the maintainable source with generated numerical inputs. Edit that template rather than the expanded `main.tex` when rerunning the pipeline.

## Reproduce the analysis

Use Python 3.13 in a fresh virtual environment. From this project directory:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r code/requirements.txt
cd code
python -m unittest discover -s tests -v
python run_all.py
python make_paper_assets.py
cd ..
```

`run_all.py` runs all simulations, historical full-data fits, genuinely training-only holdout fits, ablations, local stability analysis, numerical comparisons and the finite-horizon sensitivity map. It then generates the numerical PDF figures. It makes **no network requests**. `make_paper_assets.py` generates numerical tables/macros and expands the LaTeX template. For the static schematic, compile `code/paper_sources/scheme.tex` with PDFLaTeX and copy its PDF to `paper/scheme.pdf`. The already-compiled schematic is included. On a POSIX system with LaTeX installed, `sh build.sh` performs the complete numerical and document build.

The code also accepts `python run_all.py --part simulations`, `--part historical`, or `--part figures`. Running only `figures` requires the included numerical outputs. Main simulations use 100 stochastic replicates each; ODE comparisons use 30 configurations with 20 replicates each. Random seeds and input/source SHA-256 hashes are in `code/results/run_manifest.json`. Numeric arrays and curves are retained in `code/results/`.

## What changed

Read `audit/REVISION_NOTES.md` for the editor-comment mapping and `audit/UPSTREAM_PROVENANCE.md` for code lineage. The principal corrections are consistent annual units; complete Scenario B forcing schedules; an actual symmetric, locally stable coexistence example; event-aware ODE integration; competing-event stochastic updates; corrected historical observation definitions; continuous fitting without breakpoint resets; identifiable reduced parameterizations; temporal holdouts; and explicit limits on empirical and causal claims.

The historical fits intentionally do **not** claim to calibrate the full multi-religion, role-structured system. They are reduced descriptive benchmarks. No measured social or political covariate has been fitted. These are scientific scope boundaries, not hidden implementation options.

## Source verification boundaries

The complete Swedish percentage table was checked against its official PDF. New Zealand numerators and all-resident denominators are documented against government sources. Turkey uses a consistent peak-publisher series, not a mixture of peak and average counts. The exact scope of primary-source checks, including any retained upstream transcriptions, is explicit in `code/data/provenance.json`. The bibliography retains the user's broader reference library; targeted repairs and missing references were checked, but the package does not claim an exhaustive independent bibliographic audit of every historical book or article.

This package is ready for author review and repository upload. It has not been uploaded to GitHub, submitted to a journal, or independently peer reviewed. Do not mix its figures or tables with the old manuscript.
