# PROJECT_SPEC.md: High-Frequency L1 Queue Depletion & Point Process Econometrics

> **Durum notu:** Bu spec masa başı revizyonu tamamlamış, ancak henüz tek satır veri ile doğrulanmamıştır. Kod yazmaya başlamadan önce `scripts/00_inspect_day1.py`'yi çalıştırıp §9'daki dört sayıyı elde et. O sayılar bu spec'in bazı parametrelerini (β ızgarası, örneklem büyüklüğü, tie-break kuralı) değiştirebilir — bu beklenen ve istenen bir şeydir.

## 1. Executive Summary & Working Title

* **Working Title:** *Parametric Inductive Bias vs. Free Feature Learning: Evaluating Online Multivariate Hawkes Representations Against Multi-Scale EWMA Banks in High-Frequency L1 Liquidity Depletion*
* **Core Research Question:** Under what sample-size regimes ($N$), data-diversity conditions, and cross-instrument transfer steps does regularized, unsupervised parametric compression via Multivariate Hawkes point processes outperform unconstrained tree learning (LightGBM) on Multi-Scale EWMA feature banks when forecasting L1 queue depletion?
* **Scientific Philosophy:** Falsification-first. Single/multi-exponential Markovian Hawkes intensities are, by construction, a linear projection of an EWMA feature bank (see §2.4, $M_2$ note). The empirical objective is *not* to prove Hawkes adds information — it structurally cannot, in the raw feature sense — but to test whether MLE-constrained dimension reduction (unsupervised) generalizes better than unconstrained tree learning under small-sample, regime-shifted, or cross-instrument conditions where the tree has less to learn from.

---

## 2. Research Hypotheses & Experimental Design

### 2.1. Primary Endpoint & Target Formulations

One single **Primary Confirmatory Endpoint** is pre-registered to control FWER without multiplicity dilution:

* **Primary Endpoint:** Target A (Adverse Price Shift), $X = 50\%$, Horizon $h = 500\text{ ms}$, Buffer $\Delta = 50\text{ ms}$.
* **Preregistered Effect Size Threshold:** $\Delta\text{PR-AUC} \ge +0.005$ over $M_1$, at $\alpha = 0.05$. This threshold is fixed **before** looking at results.

**Secondary / Exploratory Grid** (reported but not confirmatory):

* **Target A (Adverse Price Shift):** Best price level moves adversely (tick down for bid, tick up for ask) within $[t+\Delta,\ t+\Delta+h]$.
* **Target B (Pure Depth Depletion):** Best price remains constant, but queue size drops $>X\%$ within $[t+\Delta,\ t+\Delta+h]$. Reported separately from Target A — do not merge into a hybrid label.
* Grid: $X \in \{30\%, 50\%, 70\%\}$, $h \in \{100\text{ms}, 500\text{ms}, 2\text{s}\}$, $\Delta = 50\text{ms}$ fixed.

### 2.2. Core Experiment 1: Controlled Learning Curve & Data Diversity

**Test set is strictly fixed** to the final 10 days (Days 51–60) across every training configuration, so all $N$ are compared on identical observations.

1. **Sample Budget Scale ($N$):** rolling backward windows ending at the test-set boundary, $N \in \{2\text{h (contiguous)}, 1\text{d}, 3\text{d}, 10\text{d}, 20\text{d}, 50\text{d}\}$.
2. **Data Diversity Control:** compare $N=2\text{h contiguous}$ against $N=2\text{h stratified}$ (sampled across 20 days) to separate sample *volume* from regime *diversity*.
3. **Objective:** locate empirical crossover $N^*$ where $M_1$ (free tree) overtakes $M_2$ (parametric Hawkes) on the fixed test set.
4. **LightGBM regularization:** tune `num_leaves`, `min_data_in_leaf`, `colsample_bytree`, `reg_lambda` per $N$ independently — do not reuse the 50-day hyperparameters for the 2-hour run, or the "trees overfit on small N" finding is an artifact of untuned regularization, not of the inductive-bias hypothesis.

### 2.3. Core Experiment 2: Dimensionless Transfer Ladder

Transfer the dimensionless branching ratio matrix $\boldsymbol{\Gamma} = \sum_p \mathbf{A}^{(p)}/\beta_p$ (not raw $\mathbf{A}$) down a relative-tick-size ladder:

$$\text{BTCUSDT} \longrightarrow \text{ETHUSDT} \longrightarrow \text{SOLUSDT} \longrightarrow \text{DOGEUSDT}$$

* **Transfer protocol:** fix $\boldsymbol{\Gamma}$ from BTC, re-estimate only $\boldsymbol{\mu}(t)$ on the target asset, scale marks $m_k$ by the target asset's own rolling median trade volume (never BTC's).
* **Why the ladder, not a single jump:** relative tick size and event-mix (e.g. frequency of `bid_price_down`) differ sharply between BTC and DOGE. Stepping through ETH → SOL isolates *where* transfer degrades rather than only whether it does.
* Report $\rho(\Gamma)$ stability and per-asset PR-AUC at each rung, not just the endpoint.

**Update (2026-09-10): all three rungs run, including the literal daisy chain and a properly profiled μ estimator — see `STATUS_TRANSFER_LADDER.md`.** M0<M2(transferred Γ)<M1 on all three targets, same qualitative story as BTC itself; the transferred Γ captures 82–89% of each asset's own M0→M1 gap, no sign of degradation down the ladder. Three design questions were each tested rather than assumed: direct-from-BTC vs. the literal chain (Δ≈0.0008–0.0009, no consistent direction), the qty-tie dual-Γ stability check (`STATUS_GAMMA_ATTRIBUTION.md` §12.4, negligible on all three assets), and empirical-rate vs. properly profiled μ (a new `profile_mu_given_alpha` estimator, Δ≤0.0003 across all 8 cases tested). None of the disclosed simplifications changed the conclusion — this is now a fully closed-out first pass at the transfer ladder.

**Update (2026-09-18):** extended to three more assets (BNBUSDT, XRPUSDT, ADAUSDT), same protocol including the profiled-μ estimator — see `STATUS_ROBUSTNESS.md` Part B. M0 < M2(transferred) < M1 still holds on all three, but all three land at 78.97%–80.27% of their own M0→M1 gap captured, below the original three-asset range's floor (81.7%) rather than inside or above it as the first pass's three rungs suggested — reported as a genuine, unsoftened finding, not folded into the "no evidence of degradation" framing above.

### 2.4. Model Ladder

* **$M_0$ (Baseline L1 LOB):** $\text{OFI}_{L1}$, micro-price basis $(P_{\text{micro}}-P_{\text{mid}})$, bid-ask spread, **absolute queue sizes and rolling historical percentile rank** $\text{Rank}(Q(t)) \in [0,1]$ (critical — omitting this makes $M_0$ artificially weak and inflates every model above it).
* **$M_1$ (Multi-Scale EWMA Bank — unrestricted tree learning):** $M_0$ + 10-event × 5-scale exponential event counts:
  $$Z_j^{(\beta_p)}(t) = \int_0^t e^{-\beta_p(t-s)}\,dN_j(s), \quad \beta_p \in \{0.2, 1.0, 5.0, 25.0, 100.0\}\ \text{s}^{-1}, \quad j \in \{1,\dots,10\}$$
* **$M_2$ (Parametric Hawkes — inductive bias benchmark):** $M_0$ + MLE-constrained intensity vector:
  $$\boldsymbol{\lambda}(t) = \boldsymbol{\mu}(t) + \sum_{p=1}^5 \mathbf{A}^{(p)}\mathbf{Z}^{(\beta_p)}(t)$$
  **Note (structural, not empirical):** with the same $\beta$ grid as $M_1$, $\boldsymbol{\lambda}(t)$ is a linear function of $M_1$'s own feature bank. $M_2$ cannot add information a tree can't already represent; it can only add a *regularization prior* over that same feature space. MLE here is unsupervised (fits event likelihood, not the forecasting target) — frame every $M_2$ vs $M_1$ comparison as "does constrained compression generalize better," never as "does Hawkes see something trees can't."
* **$M_3$ (Structural Non-Nested Dynamics):** $M_1$ + rolling hourly spectral radius $\rho(\boldsymbol{\Gamma}_t)$ (warm-started, non-overlapping, **hourly-resolution regime indicator — not expected to carry 100ms-horizon micro-timing signal**) + linearized trade-size marks.

---

## 3. Microstructure Event Taxonomy & Mathematics

### 3.1. 10-Event State Disambiguation with Residual Volume Cancellation

$$\text{Cancellation volume: } V_{\text{cancel}}(p) = \max\!\big(0,\ -\Delta Q(p) - V_{\text{trade}}(p)\big)$$

1. `bid_qty_up`: bid price unchanged, $\Delta Q_b > 0$.
2. `bid_qty_down`: bid price unchanged, $V_{\text{cancel}}(P_b) > 0$.
3. `bid_price_up`: bid price increases (spread narrowing / price improvement).
4. `bid_price_down`: bid price decreases (queue depletion / spread widening).
5. `ask_qty_up`: ask price unchanged, $\Delta Q_a > 0$.
6. `ask_qty_down`: ask price unchanged, $V_{\text{cancel}}(P_a) > 0$.
7. `ask_price_up`: ask price increases (queue depletion / spread widening).
8. `ask_price_down`: ask price decreases (spread narrowing / price improvement).
9. `trade_buy`: aggressive buyer-initiated fill.
10. `trade_sell`: aggressive seller-initiated fill.

Unit test requirement: every raw quote/trade transition must map to exactly one of these 10 types with no residual/unclassified category.

### 3.2. Linearized Marked Formulation (Preserving Log-Likelihood Concavity)

$$\lambda_i(t) = \mu_i(t) + \sum_{j=1}^{10}\sum_{p=1}^{P}\Big(\alpha_{ij}^{(p)} Z_j^{(\beta_p)}(t) + \alpha_{ij}^{\prime(p)} Z_{j,\text{mark}}^{(\beta_p)}(t)\Big)$$

where $m_k = \dfrac{\text{Volume}_k}{\text{Median(Volume)}_{[t-1\text{h},\,t]}}$ (asset-internal, rolling).

Marks enter **additively** (not as a bilinear $(1+\gamma m)\alpha$ product) specifically to preserve global concavity of the log-likelihood — do not "simplify" this back to a bilinear form during implementation.

### 3.3. Parametric Diurnal & Asymmetric Funding Jump Model

$$\mu_i(t) = \Big(a_0 + \sum_{k=1}^{8}\big(a_k\cos(\tfrac{2\pi k t_{\text{sec}}}{86400}) + b_k\sin(\tfrac{2\pi k t_{\text{sec}}}{86400})\big)\Big)\cdot \exp\Big(\sum_{f=1}^{3}\big(\delta_{\text{pre}}\mathbb{1}_{[-5\text{m},0)} + \delta_{\text{post}}\mathbb{1}_{[0,+5\text{m}]}\big)\Big)$$

$t_{\text{fund}} \in \{00{:}00,\ 08{:}00,\ 16{:}00\}$ UTC. Pre/post funding jump coefficients are asymmetric by design (funding effects are not symmetric around the event).

---

## 4. Data Pipeline & Causal Alignment Protocol

### 4.1. Data Source & Volume Budget

* **Target instrument:** `BTCUSDT` Perpetual Futures (`futures/um/daily/`).
* **Raw feeds:** `trades` (unaggregated execution ticks — not `aggTrades`, which collapses simultaneous fills and destroys the clustering signal Hawkes is meant to measure) + `bookTicker` (L1 best bid/ask updates).
* **Timestamp field:** use `transaction_time` (T) on both feeds. Do not use `event_time` (E) — it includes engine push latency.
* **Known data-quality issue #1 (ordering):** Binance's own `binance-public-data` GitHub repo (Issue #305) documents that BTC/ETH futures `bookTicker` daily files shipped **out of sequence** during Jan–Mar 2024. Do not assume the archive file is pre-sorted — sort explicitly by `(transaction_time, update_id)` before any diff-based computation.
* **Known data-quality issue #2 (discontinuation — CONFIRMED, blocks recent-date plans):** Binance stopped publishing `futures/um/daily/bookTicker/BTCUSDT` after **2024-03-30**. This is documented and still open as of Aug 2026 (`binance-public-data` Issue #372, filed Nov 2024, unresolved 20+ months later — treat as permanent, not transient). `trades` archives remain continuous and current; only `bookTicker` is affected.
* **Update (2026-09-18):** re-verified live against the S3 listing — still confirmed true, and pinned more precisely: a monthly rollup archive exists one step past the daily cutoff (`BTCUSDT-bookTicker-2024-04.zip`) but on inspection contains only 6h47m of 2024-04-01, not a full month; no genuinely new non-overlapping window is fetchable. See `STATUS_ROBUSTNESS.md` Part A for the full check and the within-window alternate-test-week robustness run that was done instead.
* **Resolved data window (supersedes the original "most recent 60 days" plan):** the study uses the **last available continuous window before the bookTicker cutoff**: **2024-01-31 through 2024-03-30** (60 calendar days) for both `trades` and `bookTicker`. Confirm the exact first/last available `bookTicker` filenames via the S3 listing (`?list-type=2&prefix=...`) before fixing exact boundary dates — do not assume calendar round numbers.
* **Limitation to state explicitly in the thesis:** results describe market microstructure during Jan–Mar 2024, not the present. This is standard for microstructure econometrics (dynamics are not claimed to be time-invariant, but the mechanisms studied — order flow clustering, cross-excitation — are not expected to have fundamentally changed) and should be stated as a scope limitation, not hidden.
* **Volume budget (measured):** sample day (2024-03-30) yields 1,469,267 `trades` rows + 7,398,592 `bookTicker` rows = 8,867,859 raw events/day. Linear 60-day estimate: **~532M raw events** (revised down from the original 800M–1.2B guess — do not use the old number). Raw CSV+ZIP footprint for the full window: ~52.6GB pre-Parquet (see §10.4 for the storage/deletion plan).

### 4.2. Timestamp Alignment & Sensitivity

* **Sub-millisecond tie-breaking:** deterministic micro-jitter $t_k^* = T_k + k\cdot 10^{-7}\text{s}$ for ties within a single feed.
* **Secondary tie-break key:** when `transaction_time` ties within `bookTicker`, use `update_id` as the ordering key before falling back to arrival order — it is a closer proxy to true sequence than an arbitrary jitter.
* **Cross-feed default ordering on exact-ms ties:** `[Trades(T) → BookTicker(T)]`.
* **Causality sensitivity switch (mandatory):** re-run the Primary Endpoint under the inverted tie rule `[BookTicker(T) → Trades(T)]`. Report whether cross-excitation parameters and the Primary Endpoint result are stable under the swap.

---

## 5. Statistical Validation & Computationally Feasible Testing

### 5.1. Minute-Averaged Giacomini-White Test Protocol

1. Compute $d_t = L(y_t,\hat y_t^{M_a}) - L(y_t,\hat y_t^{M_b})$ at event resolution.
2. Aggregate into 1-minute non-overlapping mean blocks $\bar d_\tau$ (~14,400 points over the 10-day fixed test set).
3. Run Giacomini-White (2006) CPA test on $\bar d_\tau$ with Newey-West/HAC standard errors.
4. Re-estimate models on a **rolling** basis (retrain every 24h using the preceding $N$-day window, evaluate the next day) so the GW asymptotics — which require a genuinely out-of-sample rolling forecast sequence — are valid; do not run GW on a single static train/test split.

### 5.2. Family-Wise Error Rate Control

Romano-Wolf stepdown bootstrap across the secondary grid (2 targets × 3 $X$ × 3 $h$ = 18 specifications), resampled using **day-blocks** to preserve intraday dependency structure. The Primary Endpoint (§2.1) is exempt from this correction — it is tested once, standalone, at $\alpha=0.05$.

**Update (2026-09-10): computed twice — see `STATUS_SECONDARY_GRID.md`.** A static 7-day-test-week pass found M1 beating M2 with FWER-controlled significance (adjusted p ≤ 0.0375) across the entire grid; the user then asked for the full 31-day rolling protocol (§5.1's actual specified design, 31 real day-blocks instead of 7), which **reconfirmed every specification with tighter significance** (adjusted p ≤ 0.0215) — same sign, same magnitude order throughout. Note: `target_a`'s implementation ignores the $X$ argument, so only 12 of the 18 named specifications are actually distinct (reported explicitly, not hidden).

### 5.3. Difference-in-Differences Placebo Control

* Permute event-type labels $j \in \{1,\dots,10\}$ at fixed timestamps $t_k$ (preserves timing, destroys type-specific structure).
* **Criterion:** $M_3$ is structurally valid only if $\Delta_{\text{scramble}}(M_3) - \Delta_{\text{scramble}}(M_1) > 0$ — i.e., $M_3$'s advantage over $M_1$ must itself collapse under scrambling. Do not just check that $M_3$ degrades; check that its *edge over* $M_1$ degrades, or generic volatility clustering will pass the test.
* **Secondary placebo:** lag point-process feature inputs by $\tau=10\text{s}$; verify predictive advantage decays monotonically toward zero.

---

## 6. Operational Lead-Time Metric (No Toy PnL / No Backtest)

At a fixed false-alarm budget (top 1% alert rate), report only:

1. Fraction of adverse events captured (recall at 1% FPR).
2. Median lead time: $\Delta\tau_{\text{lead}} = t_{\text{adverse\_event}} - t_{\text{alert}}$.
3. Operational delta: $\Delta\tau_{\text{lead}}(M_2) - \Delta\tau_{\text{lead}}(M_1)$.

No execution simulation, no slippage model, no strategy return. This section stays a forecasting-quality translation, not a trading study.

**Update (2026-09-10): computed — see `STATUS_OPERATIONAL.md`.** M1 and M3 are operationally close to indistinguishable at this budget (recall and lead time both ≈ identical); M2 is measurably worse on both. Consistent with §14/`STATUS_M3.md`'s other findings.

---

## 7. Computational Optimization & MLE Budget Constraints

1. **Subsampling for calibration:** fit daily $\mathbf{A}^{(p)}$ on a stratified subsample of 10M events per window (statistical power is unaffected at 1,000 parameters; L-BFGS wall-clock is not).
2. **Warm-started hourly spectral radius:** initialize each hourly $\rho(\boldsymbol{\Gamma}_t)$ fit from the $t{-}1\text{h}$ solution (≈100 iterations → <10).
3. **Numba/C++ acceleration:** vectorized $O(P)$ state recursion via `numba(nogil=True, fastmath=True)`; validate numerically against brute-force convolution (`tests/test_state_recursion.py`) before trusting any downstream result.
4. **Stationarity constraint:** enforce $\rho(\boldsymbol{\Gamma}) < 1$ during MLE (subcriticality) — check this is binding vs. slack in practice; a fit that saturates the constraint boundary needs its $\beta$ grid revisited.

---

## 8. Directory Architecture

```text
l1-microstructure-hawkes/
├── data/
│   ├── raw/                 # Downloaded 60-day trades & bookTicker ZIPs
│   ├── parquet/             # Aligned 10-event Parquet partitions
│   └── features/            # Pre-computed feature matrices
├── src/
│   ├── data_loader/
│   │   ├── downloader.py    # Binance Data Vision scraper (trades + bookTicker)
│   │   └── aligner.py       # Transaction-time aligner & 10-event classifier
│   ├── features/
│   │   ├── ofi.py           # Level-1 OFI, micro-price, queue percentile rank
│   │   ├── ewma_bank.py     # Numba-accelerated O(1) multi-scale state bank
│   │   └── seasonality.py   # 8-harmonic Fourier ToD & asymmetric funding jumps
│   ├── models/
│   │   ├── hawkes_mle.py    # Numba-accelerated L-BFGS-B marked Hawkes engine
│   │   ├── spectral.py      # Warm-started rolling hourly spectral radius tracker
│   │   └── tree_models.py   # Tuned LightGBM wrapper (per-N hyperparameters)
│   ├── validation/
│   │   ├── learning_curve.py    # Fixed-test-set learning curve manager
│   │   ├── giacomini_white.py   # Minute-block GW test engine
│   │   └── multiple_testing.py  # Day-block Romano-Wolf stepdown
│   └── utils/
│       ├── metrics.py       # PR-AUC, Brier score, HAC loss differential
│       └── operational.py   # Fixed-alarm lead-time & adverse recall metrics
├── scripts/
│   ├── 00_inspect_day1.py   # Data verification — RUN THIS FIRST, before anything else
│   ├── 01_fetch_data.py     # 60-day automated downloader
│   ├── 02_build_features.py # Feature extraction pipeline
│   ├── 03_run_learning_curve.py
│   └── 04_run_transfer_ladder.py
├── tests/
│   ├── test_10event_typing.py
│   ├── test_state_recursion.py
│   └── test_tie_breaking_causality.py
├── requirements.txt
└── PROJECT_SPEC.md
```

---

## 9. Immediate Execution Milestone: Day 1 Inspection

**Do not write general pipeline abstractions before this passes.** Run `scripts/00_inspect_day1.py` on one real day and record:

1. Exact row counts and confirmed column names for `trades` vs `bookTicker` (do not assume — print and check).
2. Whether `bookTicker` arrives pre-sorted by `transaction_time` (it may not — see §4.1). Sort defensively either way.
3. Inter-arrival time distribution of `bookTicker` updates, in particular % below 10ms (this determines whether the $\beta=100\text{s}^{-1}$ kernel scale is meaningful or below the feed's effective resolution).
4. Exact-ms collision frequency between `trades` and `bookTicker`, **normalized against the sub-10ms density** (a raw collision count without this normalization is not informative about tie-break sensitivity).
5. Empirical distribution across the 10 event types from §3.1 — confirm no transition falls outside the taxonomy.

These five numbers should be used to revisit: the $\beta$ grid (§2.4), the tie-break rule (§4.2), and the 60-day volume estimate (§4.1) — before Phase 1 is considered complete.

### `scripts/00_inspect_day1.py`

> **Not:** `DATE` aşağıda yer tutucu olarak bırakıldı. Gerçek çalıştırmadan önce §4.1'de teyit edilen kesin son `bookTicker` gününü (2024-03-30 civarı, S3 listing ile doğrulanmış) buraya yaz — 2026 tarihleri bu feed için artık çalışmıyor (Issue #372).

```python
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os

DATE = "2024-03-30"  # confirmed via S3 listing — bookTicker/BTCUSDT last available day; re-verify before use
BASE_URL = "https://data.binance.vision/data/futures/um/daily"
os.makedirs("data/raw", exist_ok=True)

for feed in ["trades", "bookTicker"]:
    fname = f"BTCUSDT-{feed}-{DATE}.zip"
    url = f"{BASE_URL}/{feed}/BTCUSDT/{fname}"
    target_zip = f"data/raw/{fname}"
    if not os.path.exists(target_zip):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, target_zip)
    with zipfile.ZipFile(target_zip, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")

# 0. Confirm headers and column names — do not assume them.
for feed in ["trades", "bookTicker"]:
    path = f"data/raw/BTCUSDT-{feed}-{DATE}.csv"
    preview = pd.read_csv(path, nrows=5)
    print(f"\n--- {feed} preview ---")
    print(preview.columns.tolist())
    print(preview.head())

df_trades = pd.read_csv(f"data/raw/BTCUSDT-trades-{DATE}.csv")
df_ticker_raw = pd.read_csv(f"data/raw/BTCUSDT-bookTicker-{DATE}.csv")

# Measure RAW out-of-order severity BEFORE sorting — the previous version of this
# script sorted first and only asserted monotonicity afterward, which cannot detect
# how disordered the file actually arrived (Issue #305: BTC/ETH bookTicker files
# shipped out of sequence during Jan-Mar 2024).
raw_diff = df_ticker_raw['transaction_time'].diff()
pct_out_of_order = (raw_diff < 0).mean() * 100
print(f"\nRaw file: {pct_out_of_order:.2f}% of consecutive rows are out of chronological order")
if pct_out_of_order > 0:
    print("  -> Confirms Issue #305. Sort is mandatory, not optional, for this window.")

# Defensive sort — see above.
df_ticker = df_ticker_raw.sort_values(['transaction_time', 'update_id']).reset_index(drop=True)

print(f"\n--- DATA INSPECTION FOR {DATE} ---")
print(f"Trades Rows: {len(df_trades):,}")
print(f"BookTicker Rows: {len(df_ticker):,}")

df_ticker['time_diff'] = df_ticker['transaction_time'].diff()
assert (df_ticker['time_diff'].dropna() >= 0).all(), \
    "Non-monotonic after sort — deeper sequence inconsistency, investigate before proceeding."

sub_10ms = (df_ticker['time_diff'] < 10).mean() * 100
print(f"BookTicker updates with <10ms spacing (post-sort): {sub_10ms:.2f}%")

trades_ts = df_trades['time'].value_counts()
ticker_ts = df_ticker['transaction_time'].value_counts()
overlap = len(set(trades_ts.index) & set(ticker_ts.index))
print(f"Exact-ms collisions: {overlap:,} ({overlap/len(ticker_ts)*100:.2f}% of quote timestamps)")
if sub_10ms > 0:
    print(f"Collision rate relative to sub-10ms density: {(overlap/len(ticker_ts))/(sub_10ms/100):.2f}x")
```

---

## 10. Decisions Resolved from §9 Measurement (2024-03-30 sample day)

Measured on the confirmed window boundary day. Do not re-litigate these without new evidence; do re-verify item 3 across additional days before Phase 1 feature extraction.

1. **$\beta$ grid — kept, reinterpreted.** BookTicker updates arrive at sub-10ms spacing 78.55% of the time, but the exact-ms collision rate with trades, normalized against that density, is only 0.10x. Most sub-10ms activity is **not** trade-triggered — it is quote flicker/requoting, not aggressive-fill-driven microtiming. The $\beta=100\text{s}^{-1}$ ($\tau=10\text{ms}$) scale is retained but must be documented as capturing **quote-update clustering**, not trade-reaction latency. Do not interpret a strong $\beta=100$ coefficient as evidence of fast reaction to trades specifically — check the event-type decomposition (§3.1) before drawing that conclusion.
2. **Causal tie-break rule — mandatory, not optional.** Exact-ms trade/bookTicker collisions are 7.99% of all quote timestamps (372,815/day on the sample day). Scaled to 60 days this is on the order of tens of millions of ambiguously-ordered events. The §4.2 causality sensitivity switch (rerun Primary Endpoint under the inverted tie rule) is confirmed necessary and must be run before trusting any cross-excitation parameter, not treated as a nice-to-have robustness check.
3. **60-day window confirmed usable, but ordering must be spot-checked further.** Issue #305 (BTC/ETH bookTicker out-of-sequence) did **not** reproduce on 2024-03-30 (0/7,398,591 raw out-of-order rows, 0 same-ms update_id regressions) — plausibly because the original report (Jan 15 2024) predates a fix, and the window here starts Jan 31 2024. **Do not generalize from one day.** Before Phase 1 feature extraction, spot-check raw ordering (the same `raw_diff < 0` measurement from `00_inspect_day1.py`, pre-sort) on at least 3 additional days spanning the window: the first day (2024-01-31), a midpoint (~2024-02-28), and one more late day. If any of those reproduce the ordering bug, the defensive sort in `aligner.py` already handles it — this check is purely to confirm blind spots don't exist, not a blocker.
4. **Disk/storage plan (superseded by §11.4 measurement):** original estimate was ~52.6GB before Parquet conversion; see §11.4 for the measured, revised figure. Do not retain both ZIP and extracted CSV after Parquet conversion — convert day-by-day and delete the CSV immediately after each day's Parquet partition is written and verified (keep the ZIPs as the re-derivable source of truth).

---

## 11. Pilot Run Findings (2024-01-31 → 2024-02-09) — BLOCKING for Full-Scale Run

The one-shot pilot executed end-to-end successfully (11/11 tests pass, real MLE convergence, real GW statistics) but surfaced findings that **must be resolved before the 60-day run proceeds.** Do not treat the pilot's $M_1$-vs-$M_2$ comparison as informative until §11.1 is resolved.

### 11.1. ~~Archive coverage gap~~ — CORRECTED: this was a feature-builder bug, not an archive gap (see §12.1)

**This subsection's original framing was wrong and is retained only for audit-trail purposes; do not act on it.** The 128/411-row feature counts for Feb 5/8 were caused by a `numpy.searchsorted` misuse against an unsorted timestamp array in the feature builder, not by missing archive data. See §12.1 for the corrected finding and §12.2 for the real, previously-unidentified problem (severe raw ordering disorder Feb 4–21).

### 11.2. Branching-ratio matrix $\boldsymbol{\Gamma}$ is unstable under the tie-break rule — blocks the transfer ladder as specified

Causality sensitivity check (N=3d): forecast PR-AUC is stable across tie rules (Δ < 0.0015), but $\boldsymbol{\Gamma}$'s relative Frobenius-norm change between `trades_first` and `book_first` is **21.95%** (max single-element change 1.95). Both fits converge (10/10), so this is not a convergence artifact — the cross-excitation structure itself is sensitive to an arbitrary ordering convention.

This directly threatens §5.3 (Cross-Instrument Transfer Ladder): the object being transferred is not stable enough, as currently estimated, to distinguish genuine structural similarity from an artifact of the tie-break convention. **Before running the transfer ladder:** investigate which cross-excitation terms (likely trade↔quote directions) drive the 22% swing, and either (a) revise the tie-break/attribution logic so $\Gamma$ stabilizes, or (b) report $\Gamma$ transfer results only with explicit tie-rule sensitivity bands rather than a point estimate.

### 11.3. Seasonality/funding baseline not yet wired into the Hawkes MLE objective

`seasonality.py` produces the 8-harmonic Fourier and funding-jump regressors and writes them to the feature Parquet, but the pilot Hawkes fit used a constant $\mu_i$ — §4.2's baseline model has not yet been tested. Must be connected before the full run, particularly given the gap-affected days interact with funding-time windows.

### 11.4. Storage estimate revised (favorably)

Measured Parquet compression: 3.49GB raw CSV/day-pair → 258MB Parquet (~13.5x). Scaling the 10-day aligned Parquet footprint (1.86GB) to 60 days suggests **~11–12GB** for the full aligned dataset — well under the original ~52.6GB CSV-based estimate. Retain the day-by-day convert-then-delete-CSV policy regardless; the ZIP archives (~6GB) remain the source of truth.

### 11.5. Local file sync reminder

The `PROJECT_SPEC.md` copy used by the pilot run's Claude Code session was the pre-§10 version (stale). This version (§1-§13) should replace the local repo copy before the next session, to avoid re-litigating closed decisions or losing the audit trail above.

---

## 12. Full State Audit Findings (2024-09-02) — Corrects §11.1, Supersedes Its Data-Quality Narrative

A full repo/data audit (`STATE_AUDIT.md`) reconciled the §11.1 discrepancy and surfaced a more serious, previously unidentified problem. **Do not trust any Hawkes fit spanning Feb 4–21, 2024 until §12.2 is resolved.**

### 12.1. Coverage gap claim corrected: real archive coverage is excellent

The 128/411-row Feb 5/8 feature counts were a **feature-builder bug** (`numpy.searchsorted` applied to an unsorted timestamp array), not an archive gap. The corrected, full 60-day coverage audit shows **58/60 days CLEAN**, with only two genuinely gapped days: 2024-02-28 (2,587s inactive) and 2024-03-08 (3,972s inactive), totaling ~2h10m of real inactivity across the entire window. This is a *much* better data-quality picture than §11.1 assumed. Stale contaminated feature artifacts for 2024-02-04 through 2024-02-09 remain on disk under `data/features/` and `data/features_book_first/` and must be regenerated with the fixed builder (only Jan 31–Feb 4 has been regenerated clean, under `data/features_clean/`) before those days are used in any experiment.

### 12.2. Severe raw ordering disorder, Feb 4–21, 2024 — the actual blocking data-quality issue

Full-window audit of raw (pre-sort) `transaction_time` decreases shows the true problem: on multiple days in this range, up to **~40% of a day's rows** exhibit a timestamp decrease relative to the previous row (e.g., 18,728,015 decreasing-timestamp rows out of 46,396,761 total on 2024-02-12). This is Issue #305 reproducing at a scale far beyond isolated ties — this is not "a few same-millisecond rows need a tiebreak," it is large-scale scrambling of row order within the file. The defensive sort (`transaction_time`, then `update_id`) produces *a* valid-looking chronological order, but whether it recovers the *true* order at this scale of disorder has not been verified. **Required before trusting any result touching this date range:** an independent cross-check of recovered order against a source not subject to the same corruption (e.g., compare recovered inter-event spacing distributions against a known-clean day, or cross-validate against `trades` timestamps in the same window, where available — see §12.3 for `trades` coverage gaps in this exact range).

### 12.3. `trades` archive is incomplete across the 60-day window — blocks the full run independently of data-quality questions

Of 60 days: **37 fully downloaded**, **6 partial** (`.zip.part` remnants: 02-12, 02-22, 02-24, 03-09, 03-10, 03-24), **17 fully missing** (02-13, 02-14, 02-25–02-29, 03-11–03-15, 03-25–03-29). This is an incomplete download, not a source-side gap, and must simply be completed before the 60-day run — independent of and in addition to §12.2.

### 12.4. Γ instability is now localized and explained

The 21.95% relative Frobenius-norm change (§11.2) is concentrated: **86.88% of the total squared change comes from the quote→trade block alone**. The two largest single-element changes are `trade_sell ← bid_price_down` (Δ=1.953) and `trade_buy ← ask_price_up` (Δ=1.821) — both correspond to same-millisecond touch-crossing/trade pairs, where the tie rule directly determines whether the price move or the trade is treated as causally prior. This is a precise, physically interpretable target for a fix, not a diffuse instability.

### 12.5. Stationarity implementation deviates from spec

§7 specifies enforcing $\rho(\boldsymbol{\Gamma})<1$ *during* MLE. The implemented version instead runs an unconstrained fit and applies a post-hoc radial projection to $\rho<0.995$ if needed. This is a materially different estimator (the optimizer itself is unconstrained) and requires an explicit decision: accept the post-hoc projection, or require an in-fit constraint as originally specified.

### 12.6. GW test has not been re-run on clean data

The only GW result on record (§5.2 of the audit: statistic −5.86, p=4.67e-9) is computed on the contaminated Feb 5–9 features and is not valid evidence. No GW result yet exists for the clean Jan 31–Feb 4 rerun.

### 12.7. Test suite grew without spec sync

Current suite is 13/13 passing (two tests added since §11 was written: seasonal-baseline profiling, gamma-block diagnostic). §11's "11/11" is stale text, not a regression.

---

## 13. Extended Clean Window (38 days) — Design and Rationale

**This section supersedes the March-only (30-day) plan below where they conflict; the March-only subsections are retained as the immediate predecessor design, not as an alternative to choose between.**

Verification (2026-09-02) confirmed the three "missing" `trades` days initially reported near the end of February (2024-02-25, 02-26) and mid-March (03-11) were **local download artifacts, not source-side gaps** — all three downloaded successfully (HTTP 200, valid ZIPs, 13.9MB/38.0MB/56.8MB respectively) from Binance's archive. This resolves the ambiguity that previously forced a conservative 30-day, March-only window.

Combined with §12.2's finding that raw bookTicker ordering disorder ends precisely at 2024-02-21 (2024-02-22 onward shows zero raw time decreases), the primary window is extended to:

**2024-02-22 → 2024-03-30 (38 calendar days).**

This gains 8 additional days of training budget over the March-only plan at no additional risk: the only known defects in this range are the two already-characterized small gaps (2024-02-28, 2,587s; 2024-03-08, 3,972s; §12.1/§11.4), handled by the existing masking strategy. The unresolved Feb 4–21 disorder (§12.2) remains entirely outside this window and is still not attempted here.

**Before finalizing:** complete the `trades` downloads for all days in 2024-02-22 → 2024-03-30 that are not yet fully present locally (the three verified-available days above, plus any other partial/missing days in this specific sub-range — re-check against the full inventory in §12.3, which was computed over the original 60-day window and needs to be re-filtered to this 38-day range specifically).

### 13.0. Fixed test set and $N$ grid, updated for 38 days

§13.1's test-set rationale (representativeness over raw power, full-week coverage) is unchanged. With 38 days and a 7-day fixed test set, maximum training span becomes **31 days**. Adjusted grid: $N \in \{2h, 1d, 3d, 7d, 14d, 23d, 31d\}$ — this restores meaningful coverage toward the original spec's $N=20$–$30$ range that the March-only plan had compressed.

The test-set anchor (last 7 calendar days, 2024-03-24 → 2024-03-30) is unchanged, including the still-outstanding anomaly check from §13.1.

### 13.5. Fixed test-week anomaly check (completed)

The fixed 2024-03-24 → 2024-03-30 test week was compared with the 2024-02-22 → 2024-03-23 pre-test days using per-day robust z-scores, $z=(x-\mathrm{median})/(1.4826\,\mathrm{MAD})$, with an ex-ante flag threshold of $|z|>3.5$. The audited metrics were trade row count, quote notional, realized volatility, bookTicker row count, and aligned event count. **No test-week observation was flagged** (`flags=[]`, `test_week_anomalous=false`). The largest absolute robust z-score was 2.424538 for aligned-event count on 2024-03-30.

| Date | Trade rows z | Quote notional z | Realized-vol z | Book rows z | Aligned events z |
|---|---:|---:|---:|---:|---:|
| 2024-03-24 | -0.485300 | -0.396836 | -0.099715 | -1.344401 | -1.389692 |
| 2024-03-25 | 0.265774 | 0.284032 | 0.159993 | -0.948914 | -0.793684 |
| 2024-03-26 | -0.177222 | -0.052795 | 0.183083 | -1.236453 | -1.198864 |
| 2024-03-27 | 0.212422 | 0.360140 | 1.084375 | -1.041276 | -0.893716 |
| 2024-03-28 | -0.478587 | -0.329609 | -0.322033 | -1.337560 | -1.387352 |
| 2024-03-29 | -0.785298 | -0.568514 | -0.476675 | -1.764831 | -1.896550 |
| 2024-03-30 | -1.338192 | -0.978623 | -0.934963 | -2.148673 | -2.424538 |

---

## 13-PRIOR. March-Only Pilot Window (30 days) — Superseded, Retained for Audit Trail

**Do not use this section's window (2024-03-01 → 2024-03-30) or $N$ grid — see §13 above. Retained only so the reasoning that led to the 38-day extension is traceable.**

### 13-PRIOR.1. Fixed test set: last 7 calendar days (2024-03-24 → 2024-03-30)

Two distinct considerations govern test-set size for the learning-curve experiment (§2.2), and they point in different directions:

- **Statistical power saturates well below 7 days.** A single clean day yields on the order of 80,000+ filtered decision points (per the 10-day pilot's clean-day counts); a multi-day test set of even 2–3 days already provides ample power for stable PR-AUC and GW estimates. Raw statistical power is *not* the binding constraint on test-set size here.
- **Representativeness is the binding constraint.** The Feb pilot's central failure (§11.1, corrected but instructive) was a fixed test set landing entirely inside an anomalous period, silently biasing the M1-vs-M2 comparison. A test set shorter than 7 days systematically excludes some weekdays and risks weekday/weekend imbalance; 7 days is the minimum span that guarantees exactly one full calendar week (all weekdays, one weekend, three full 8-hour funding cycles per day) regardless of anchor point.

2024-03-24 is a Sunday, so 2024-03-24 → 2024-03-30 is a clean Sun–Sat week. This test-set anchor and its rationale carry over unchanged into §13 above.

**Required check (still outstanding, not superseded by the window extension):** verify 2024-03-24 → 2024-03-30 is not anomalous relative to the rest of the (now 38-day) window on volume, volatility, or event-rate grounds. If it is anomalous, either select a different clean full week within the window as the test set, or report the primary endpoint on two candidate weeks and check directional consistency.

### 13-PRIOR.2. Adjusted learning-curve $N$ grid (superseded by §13.0's 38-day grid)

The original $N \in \{2h,1d,3d,10d,20d,50d\}$ (§2.2) does not fit a 30-day window with a 7-day fixed test set (max available training span is 23 days). This 30-day-scoped grid ($N \in \{2h, 1d, 3d, 7d, 14d, 23d\}$) is superseded by §13.0's 38-day grid; retained only for audit trail.

### 13-PRIOR.3. Two distinct evaluation protocols — do not conflate (still applies)

This project runs two separate statistical evaluations that must not be treated as sharing one "test set":

1. **Learning-curve PR-AUC comparison (§2.2, §13.1):** one *fixed* 7-day test set, compared across all $N$ values for an apples-to-apples read on the crossover.
2. **Giacomini-White test (§5.1):** a *rolling* re-estimation scheme (daily retrain on a trailing window, forecast the next day), producing on the order of 15–20 distinct OOS evaluation days across the 30-day window — a separately powered, walk-forward assessment, independent of the fixed 7-day window's boundaries.

Both use the same underlying data but answer different questions (a static crossover point vs. walk-forward predictive ability over time); results from one should not be described as validating or invalidating the other.

---

---

## 14. 38-Day Pilot Results

All results in this section use the independently re-audited 2024-02-22 → 2024-03-30 window. The fixed learning-curve test set is 2024-03-24 → 2024-03-30 (579,593 rows; target prevalence 0.3440345208). Hawkes calibration uses at most 100,000 type-stratified events, the seasonal-profile baseline, and the existing post-hoc stationarity projection implementation described in §12.5. The projection did not bind in any static learning-curve fit (`stationarity_scale=1.0` throughout).

### 14.1. Fixed-test learning curve

| N | Train rows | M0 PR-AUC | M1 PR-AUC | M2 PR-AUC | M0 Brier | M1 Brier | M2 Brier | $\rho(\Gamma)$ | Seasonal log-likelihood improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2h | 7,200 | 0.5705600745 | 0.6869820960 | 0.6559263216 | 0.2223043455 | 0.1916784603 | 0.2098252854 | 0.9382741897 | 1,027.9521 |
| 1d | 82,799 | 0.5959138042 | 0.7008203642 | 0.6794294650 | 0.2083216397 | 0.1841098182 | 0.1873945943 | 0.9327609059 | 22,234.6526 |
| 3d | 248,397 | 0.6252981828 | 0.7050007357 | 0.6923842563 | 0.1891395145 | 0.1680171744 | 0.1707870065 | 0.9542078309 | 62,489.1357 |
| 7d | 579,593 | 0.6276258471 | 0.7058287032 | 0.6918090742 | 0.1858112027 | 0.1653064068 | 0.1685688648 | 0.9549810294 | 168,053.4852 |
| 14d | 1,159,186 | 0.6295886303 | 0.7066682758 | 0.6932490467 | 0.1911223953 | 0.1652326226 | 0.1685409388 | 0.9604496341 | 426,692.3605 |
| 23d | 1,899,085 | 0.6296402443 | 0.7062285858 | 0.6913830866 | 0.2001300809 | 0.1673230436 | 0.1723540581 | 0.9452233127 | 803,927.1825 |
| 31d | 2,558,894 | 0.6282185486 | 0.7058569833 | 0.6921162539 | 0.2139888824 | 0.1710036128 | 0.1756603864 | 0.9561896057 | 992,502.0952 |

All ten Hawkes target rows converged for every N; each fit selected 100,000 calibration events.

### 14.2. Rolling Giacomini–White test

The rolling protocol used a trailing seven-day training window and daily re-estimation from 2024-02-29 through 2024-03-30, independently of the fixed test set. It produced 31 OOS days, 2,561,477 OOS rows, and 42,694 one-minute loss blocks. With loss differential defined as M1 log loss minus M2 log loss: mean differential = **-0.01000890845**, GW statistic = **-38.58596988**, computed two-sided p-value = **0.0** (floating-point underflow at the reported precision). The negative sign favors M1.

### 14.3. Tie-break causality sensitivity (N=3d)

The inverted `book_first` rule produced M2 PR-AUC 0.6805835954 versus 0.6923842563 under `trades_first`, a difference of **-0.01180066091**. The branching-ratio matrices changed by **43.67387119%** in relative Frobenius norm; maximum absolute element change was 1.7387022694. Both fits converged 10/10. Spectral radii were 0.9544454177 (`trades_first`) and 0.9751969490 (`book_first`).

| $\Gamma$ block | Default Frobenius | Delta Frobenius | Relative change | Share of total squared change | Max absolute change |
|---|---:|---:|---:|---:|---:|
| quote→quote | 5.6551032868 | 0.7640451026 | 0.1351071879 | 0.0910755582 | 0.4557211769 |
| trade→quote | 0.2072113505 | 0.1100417658 | 0.5310605118 | 0.0018892044 | 0.0652666347 |
| quote→trade | 0.0367627312 | 2.3913246845 | 65.0475251818 | 0.8921562920 | 1.7387022694 |
| trade→trade | 1.2568443986 | 0.3088190980 | 0.2457098893 | 0.0148789454 | 0.2321324548 |

The largest changes were `trade_buy ← ask_price_up` (1.7387022694) and `trade_sell ← bid_price_down` (1.6369919305), both in the quote→trade block. A non-integrated exact-ms touch-crossing attribution proposal is retained separately under `proposals/gamma_touch_crossing_attribution.py`; it was not applied to any result above.

---

## 15. Calibration budget was fixed at 100,000 events regardless of $N$ (new session, 2026-09-08)

§7.1 specifies a 10M-event calibration subsample "at 1,000 parameters"; the
implementation actually run for every result in §14 and in `STATUS_M3.md`
used a fixed 100,000-event budget, identical at every $N$ from 2h to 31d —
i.e. not scaled with the sample-budget axis §2.2 itself varies. Full
investigation, per-$N$ raw-event-availability table, and a single N=14d
control fit at 10× the budget (which moved M2 by a real, above-noise
amount but left M3's near-null Primary Endpoint result unchanged, for a
structural reason specific to M3's feature set) are in
`STATUS_CALIBRATION_BUDGET.md`. §14's learning curve and §5.1 of
`STATUS_M3.md` are not superseded by this finding, but §14's exact
crossover-point read is now flagged as unreliable until re-run with an
$N$-scaled calibration budget — see that document's §4 for the recommended
design. Not re-run here.

**Update (2026-09-09/10): the re-run happened — see `STATUS_CALIBRATION_BUDGET.md`
§5.** A full N-scaled sweep, controlled against a same-machine flat-100k
rerun (a machine-reproducibility finding made this necessary — see §5.1
of that document), found a real, positive ΔM2 at every N, closing the
M1−M2 gap by 3.7%–20.0% depending on N. **§14's headline finding is
confirmed robust, not just "unlikely to reverse": M1 leads M2 at every N
under either budget, by a full order of magnitude above the pre-registered
threshold even at the most aggressive budget tested (31d, 5,000,000
events). No crossover N\* exists.** What §14's table gets wrong is the
*exact magnitude* of the gap, not its direction.

---

## 16. Per-N LightGBM tuning — the standing "M1 is undertuned" hypothesis was backwards (2026-09-10)

A caveat carried since §14 was written speculated that M1's hyperparameters
being undertuned relative to M2's (both use the same coarse 3-bucket
heuristic) understates M1's true advantage. **Tested — see
`STATUS_PER_N_TUNING.md`. It's backwards: real per-N tuning of both
models independently narrows the M1−M2 gap at every N from 3d to 31d (by
11–19%), because M2 gains more from added tree complexity than M1 does.**
The qualitative conclusion (M1 > M2 everywhere) is unaffected — the gap
narrows, it does not close. N=2h/1d tuning did not generalize from
validation to the test week (a small-sample caveat specific to those two
points, not a competing finding) — see that document's §4.
