#!/bin/sh
set -eu
cd "$(dirname "$0")"
ROOT=$(pwd)
(cd code && python -m unittest discover -s tests -v && python run_all.py && python make_paper_assets.py)
(cd code/paper_sources && pdflatex -interaction=nonstopmode -halt-on-error scheme.tex)
cp code/paper_sources/scheme.pdf paper/scheme.pdf
cp code/paper_sources/scheme.pdf code/results/figures/scheme.pdf
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
