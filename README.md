# The Horizon Transform: An Exactly Invertible, Linear-Time Multi-Scale Layer for Long Sequences

Jagadeesh Selvaraj, NextGen Technologies Private Limited, Chennai, India
Email: dr.jagadeeshselvaraj@nextgentpl.com | ORCID: https://orcid.org/0009-0007-2314-4057 | GitHub: https://github.com/drjagadeeshselvaraj

This repository holds the journal article, its supplementary materials and the code and result files behind every number in them. The Horizon Transform (HT) is a causal bank of damped complex oscillators with geometrically spaced horizons, and HT-A is its input-adaptive variant.

**Scope.** All experiments are reduced-scale: synthetic data, numpy on one CPU core, no trained models. The article states the positive results and, in its Section 8, the findings in which the default bank is not competitive and the work that remains.

## Layout

| Folder | Contents |
|---|---|
| `article/` | Journal article (`.docx` with editable equations, `.pdf` with equations rendered as images), the 12 figures it uses, and the scripts that build it |
| `supplementary/` | Supplementary Materials (`.docx`, `.pdf`): reproducibility guide, task definitions, bank parameters, full numerical results, notation, figure and table index, change log; and the script that builds it |
| `code/` | `lib.py` (banks, memory probes, ridge readout), experiment scripts `e0.py` to `e16d.py`, figure scripts `figs*.py`, result files `out/*.json`, figures `out/*.png`, `requirements.txt`, and `tools/` (document-building helpers) |

## Quick start

```
cd code
pip install -r requirements.txt
python3 e2.py          # memory probes and ablations
python3 e14.py         # orthogonalized bank and hybrids
python3 figs_ch9.py    # figures for Chapter 9 of the book
```

Each experiment script writes its results to `out/` and the figure scripts read them from there. `code/README.md` lists every script with its purpose, the sections it supports and its approximate run time; the supplementary materials (Part S2) give the seeds and the exact task definitions. The timing scripts (`e5.py`, `e13.py`) depend on the machine and should be run on an idle CPU.

## Notes

- The scripts in `article/`, `supplementary/` and `code/tools/` that build documents use fixed paths (`/home/claude/...` and `/mnt/user-data/outputs`) and must be adapted to run elsewhere; the experiment and figure scripts use relative paths.
- Of the 61 references in the article, only six were checked against web sources during preparation; the SWIFT reference lacks its author list. Verify the references before submission.
- License: not yet chosen.
- Citation metadata: `CITATION.cff`.
