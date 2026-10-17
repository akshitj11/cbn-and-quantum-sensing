# arXiv note (draft)

Not submitted. This is the LaTeX recast of the magnetometer writeup.
You still need an endorsement (or a prior paper in the same arXiv domain)
before a first upload. Categories that fit: `quant-ph` and `cs.LG`.

```bash
cd docs/arxiv
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Source: `main.tex`, `refs.bib`.
The informal version is `../clifford-magnetometer-whitepaper.md`.
