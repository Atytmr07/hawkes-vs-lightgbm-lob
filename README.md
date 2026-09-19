# Parametric Inductive Bias vs. Free Feature Learning

**Evaluating a multivariate Hawkes self-exciting point process against gradient-boosted
trees for L1 limit-order-book queue depletion forecasting.**

Pre-thesis empirical research project. BTCUSDT Binance perpetual futures, 38-day
window, 1.05B raw microstructure events.

## Research question

Under what sample-size, data-diversity, and cross-instrument transfer conditions does a
regularized, unsupervised, MLE-constrained multivariate Hawkes process ($M_2$)
generalize better than an unconstrained LightGBM gradient-boosted-tree model ($M_1$),
given that $M_2$'s intensity is, by construction, a linear projection of $M_1$'s own
feature space and therefore cannot see information $M_1$ cannot already represent? The
project is falsification-first: it does not set out to prove Hawkes processes add
information, but to test whether parametric compression is a better regularizer than
free tree learning under small-sample or cross-instrument conditions.

## Headline findings

- $M_1$ (free tree) beats $M_2$ (Hawkes) at every training-sample budget tested
  (2h-31d), by 0.011-0.031 PR-AUC -- 2-6x the pre-registered +0.005 significance
  threshold, with no crossover sample size. Confirmed robust under a 50x calibration
  budget increase, an 18-specification Romano-Wolf FWER-controlled test
  (adjusted $p \le 0.0215$), and an independent held-out test week.
- $M_3$ ($M_1$ + a Hawkes-derived hourly regime indicator) shows no PR-AUC advantage
  over $M_1$, but a formal rolling Giacomini-White test finds a small, highly
  significant *calibration* (log-loss) improvement ($p \approx 5.1\times10^{-12}$;
  regime-only variant, $p \approx 2.3\times10^{-15}$), traced by ablation to the regime
  feature, not the trade-size marks.
- The BTC-fitted branching-ratio matrix $\Gamma$, transferred to six other assets
  (ETH, SOL, DOGE, BNB, XRP, ADA) with only the baseline intensity re-profiled, retains
  79-89% of each target's own baseline-to-tree predictive gap -- with a disclosed,
  unresolved split between the original three assets and the three added under a later
  robustness pass.
- A standing project hypothesis that undertuned LightGBM hyperparameters understated
  $M_1$'s advantage was tested directly and found backwards: proper per-N tuning
  *narrows* the $M_1$-$M_2$ gap by 11-19%, not widens it.

Full numeric results, including every surprising or null finding, are in `paper/`
(arXiv-style write-up); `PROJECT_SPEC.md` records the research design and protocol
decisions behind them.

## Repository layout

```
PROJECT_SPEC.md        research design, hypotheses, data pipeline, protocol decisions
RELATED_WORK.md          literature review, references.bib has the BibTeX
paper/                  arXiv-style LaTeX paper (main.tex, sections/)
src/
  data_loader/          Binance archive download + causal trade/quote alignment
  features/              OFI/microprice (M0), multi-scale EWMA bank (M1), marks (M3)
  models/                 Hawkes MLE engine (M2), rolling spectral radius (M3),
                          LightGBM wrapper + tuning
  validation/            learning-curve orchestration, Giacomini-White test,
                          Romano-Wolf stepdown
  utils/                  metrics (PR-AUC, Brier, Newey-West HAC)
scripts/                one script per experiment/analysis run
tests/                  pytest suite (29 tests at last count)
```

`data/` (raw/aligned/feature Parquet, tens of GB, regenerated via
`scripts/01_fetch_data.py` + `scripts/02_build_features.py`) is intentionally not
versioned here -- see `.gitignore`.

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest -q
```

Individual experiments are one script each under `scripts/`; `paper/sections/results.tex`
and `paper/sections/methodology.tex` document which script and parameters produced each
reported number.

## Status

Empirical findings are complete and documented; the project is currently moving into
thesis draft-writing, with faculty advisor review to follow. Methodology, findings, and
their exact provenance are transparently recorded as they happened, including corrected
prior assumptions -- this project's own convention is to report unexpected or negative
results plainly rather than omit them.

## License

Licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) --
see `LICENSE`. You may read and share this work with attribution; commercial use and
distribution of modified versions are not permitted. If you want to build on this work,
please reach out rather than forking a modified copy.

## Data source note

Built on Binance Data Vision (`data.binance.vision`) public archives. Only code is
distributed here; raw exchange data is not redistributed and must be re-fetched per
Binance's own terms via the scripts in this repository.
