# Horizon Transform experiments

Scripts and raw results behind Chapter 8 of Horizon_Transform.docx. Requires Python 3 with numpy, scipy, matplotlib, scikit-learn, Pillow. No GPU, no network, no deep-learning framework. All runs were on one CPU core.

Run from this folder (results go to `out/`):

| Script | What it does | Book section | Approx. time |
|---|---|---|---|
| `e0.py` | Property checks (recurrence, inverse, variance formula, stability bound, scan, HT-A invertibility probe) | 8.2 | seconds |
| `e1.py` | Reconstruction under noise and quantization, incl. HT tuning | 8.3 | ~25 s |
| `e2.py` | Memory capacity, scale-free recall, free-pole tuning, ablations | 8.4, 8.8 | ~4 min |
| `e3.py` | Forecasting and spectral classification | 8.5 | ~70 s |
| `e4.py` | Selective recall (HT, HT-A, selective real-pole SSM) | 8.6 | ~5 min |
| `e5.py` | Latency and peak memory against attention (run on an otherwise idle CPU) | 8.7 | ~2 min |
| `e6.py` | Decoder comparison: pooled, noise-weighted pooled, least squares, weighted least squares (needs `e1.py` results) | 4.5 | ~10 s |
| `e7.py` | Precision of the exact inverse, erasure robustness, effective channel count and least-squares symbol | 4.1, 4.3, 4.4, 4.6 | ~15 s |
| `e8.py` | Cost scaling, group delay, stability bound, HT-A boundedness, frame bounds, gradient check, dilation equivariance | 5.1 to 5.7 | ~20 s |
| `e9.py` | Relations to other transforms: DFT limit, exponential-window STFT equivalence, quality factors, gammatone cascade, S4D-Lin mapping, mixing matrices | 6.2 to 6.7 | ~10 s |
| `e10.py` | Statistical planning: minimal detectable effects, retrospective application of the decision rules to H1 and H2, run counts (needs `e1.py` and `e4.py` results) | 7.6 to 7.10 | ~20 s |
| `e11a.py` | Per-seed paired memory comparisons with Holm correction; sensitivity to observation noise | 8.11, 8.12 | ~30 s |
| `e11b.py` | Feature-count sweep for the main banks | 8.12 | ~15 s |
| `e12.py` | Classification accuracy against task difficulty | 8.12 | ~2 min |
| `e13.py` | Latency and peak memory with five repetitions at every length (run on an idle CPU) | 8.13 | ~1 min |
| `e14.py` | Mitigation experiments: orthogonalized HT bank and HT + exact-recall hybrids on the memory probes | 9.6 | ~25 s |
| `e15a.py` | Limits: nonlinear targets with linear and second-order readouts; storage precision of the orthogonalized bank | 10.2, 10.5 | ~40 s |
| `e15b.py` | Memory probes for four input distributions | 10.4 | ~30 s |
| `e15c.py` | True float32 recursion for the default and orthogonalized banks | 10.5 | ~10 s |
| `e16a.py` | Pilot: orthogonalized bank on reconstruction, forecasting and classification | 11.7 | ~30 s |
| `e16b.py` | Pilot: scales against frequencies per scale, default and orthogonalized bank | 11.8 | ~15 s |
| `e16c.py` | Pilot: gammatone-style cascades of order 1 to 4 at equal delay | 11.9 | ~40 s |
| `e16d.py` | Pilot: irregular sampling (naive, zero-filled, time-aware) | 11.10 | ~80 s |
| `figs.py`, `figs2.py`, `figs_ch1.py`, `figs_ch2.py`, `figs_ch3.py`, `figs_ch4.py`, `figs_ch5.py`, `figs_ch6.py`, `figs_ch7.py`, `figs_ch8x.py`, `figs_ch9.py`, `figs_ch10.py`, `figs_ch11.py`, `figs_ch12.py` | Figures 1 to 8 from the JSON results; schematic and computed figures for Chapters 1 to 12 | all | seconds |

`lib.py` holds the shared HT parametrization, feature banks and scoring functions.

Notes
- Seeds: stochastic parts use seeds 0 to 4; data seeds are fixed in each script, so reruns reproduce the reported numbers up to timing noise in `e5.py`.
- `e1.py` also computes a Morlet CWT reconstruction bank. It is not a stable frame at 8 scales or fewer and is excluded from the report; its entries remain in `out/e1.json`.
- "HT default" uses the frequency rule omega = pi (j+1) / (2 h) with two frequencies per horizon scale. This rule is a choice made for these tests (see `lib.ht_default`).
- Scale-free recall (`lib.scale_free_recall`) and the selective-recall task in `e4.py` were designed for this report and favour the HT premise.
- Not run: Long Range Arena, standard forecasting datasets, audio or EEG data, trained Transformer, Mamba or S4 models.


## tools/
`bookkit.py` (helpers that write formatted paragraphs, tables, figures and the contents into the book’s docx) and `ch9_apply.py` (inserts the Chapter 9 extension). They edit the finished docx in place; the earlier build scripts for Chapters 1 to 8 are not included.

The Chapter 10 extension was applied to the book once with the same helpers (`tools/bookkit.py`); its insertion script is not included.

`kdp_convert.py`, `kdp_pdf.py` and `eq2png.py` in `tools/` build the 8.5 x 11 in KDP edition (Word file and print PDF) from the finished book. They read the book from the outputs folder and write to /home/claude/kdp.

`sup_build.py` in `tools/` builds the Supplementary Materials document from the result files and the book.

`article_build.py` and `article_pdf.py` in `tools/` build the journal-article version of the book (Word file with editable equations, and a PDF with equations rendered as images).
