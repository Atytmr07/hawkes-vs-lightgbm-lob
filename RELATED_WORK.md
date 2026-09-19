# Related Work

Compiled via live web search (September 2026) to ground this project's Hawkes-vs-LightGBM L1
queue-depletion study in real, verifiable literature. **Every entry below was confirmed to exist**
through at least one independent source (arXiv abstract page, publisher page, SSRN, or
Semantic Scholar/ResearchGate listing) before being included here. Where a bibliographic detail
(exact page range, exact venue, or peer-review status) could not be independently confirmed, this
is stated explicitly next to the entry rather than guessed. BibTeX entries for all 21 papers are
in `references.bib`.

---

## (a) Hawkes processes / self-exciting point processes in finance and market microstructure

1. **Bacry, Mastromatteo & Muzy (2015), "Hawkes Processes in Finance"**, *Market Microstructure
   and Liquidity* 1(1):1550005; arXiv:1502.04592.
   The standard survey of Hawkes applications in finance (volatility estimation, market
   stability, systemic contagion, optimal execution, full order-book dynamics). This project's
   10-event marked multivariate Hawkes with exponential kernels and a branching-ratio
   subcriticality constraint is a direct descendant of the framework this survey organizes —
   it is the natural §1/§2 citation for *why* Hawkes processes are the chosen parametric family.

2. **Hardiman, Bercot & Bouchaud (2013), "Critical reflexivity in financial markets: a Hawkes
   process analysis"**, *European Physical Journal B* 86:442; arXiv:1302.1405.
   Finds price-change Hawkes kernels are power-law and close to criticality (branching ratio
   ≈1) across 1998–2011 S&P E-mini data. Directly relevant context for this project's spectral-
   radius subcriticality enforcement (ρ(Γ)<1): the project's measured ρ(Γ) values (0.93–0.98,
   §14.1 of PROJECT_SPEC.md) sit in the same near-critical regime this paper documents as a
   persistent stylized fact, not an estimation artifact.

3. **Filimonov & Sornette (2015), "Apparent criticality and calibration issues in the Hawkes
   self-excited point process model: application to high-frequency financial data"**,
   *Quantitative Finance* 15(8); arXiv:1308.6756.
   Warns that regime mixtures and poor timestamp quality can produce spuriously high estimated
   branching ratios (apparent criticality where the true value is far lower). This is the
   methodological counterpoint the project should cite alongside its own §11.2/§12.4 findings —
   the tie-break-rule sensitivity of Γ (21.95%–43.67% relative Frobenius-norm swings under
   `trades_first` vs `book_first`) is exactly the kind of calibration fragility this paper warns
   about, and the project's decision to report tie-rule sensitivity bands rather than a single Γ
   point estimate follows its spirit.

4. **Bowsher (2007), "Modelling security market events in continuous time: Intensity based,
   multivariate point process models"**, *Journal of Econometrics* 141(2):876–912.
   Introduces "generalised Hawkes" (g-Hawkes) models allowing inhibitory effects and cross-day
   dependence, plus specification tests via random time change. Foundational for treating market
   events (trades, quote changes) as a multivariate point process with a full intensity vector —
   the same modeling stance this project takes with its 10-event taxonomy, extended with
   explicit event-type cross-excitation.

5. **Bormetti, Calcagnile, Treccani, Corsi, Marmi & Lillo (2015), "Modelling systemic price
   cojumps with Hawkes factor models"**, *Quantitative Finance* 15(7).
   Uses multivariate Hawkes factor models to capture contagion/cojumps across correlated
   assets. Relevant as precedent for this project's cross-asset transfer ladder
   (BTC→ETH→SOL→DOGE, STATUS_TRANSFER_LADDER.md) — both projects use a shared/constrained Hawkes
   structure to transport self-/cross-excitation information across instruments rather than
   re-estimating independently.

6. **Hewlett (2006), "Clustering of order arrivals, price impact and trade path optimisation"**,
   Workshop on Financial Modeling with Jump Processes, Ecole Polytechnique.
   *Flag: this is a workshop presentation, not a peer-reviewed journal article — cite it as such.*
   One of the earliest applications of a bivariate Hawkes process to buy/sell trade arrivals in
   FX, used to predict imbalance and derive market-maker pricing. Relevant precedent for treating
   trade-side imbalance as a Hawkes-predictable quantity, a narrower ancestor of this project's
   10-event taxonomy.

## (b) Limit order book modeling and queue-position / queue-depletion prediction

7. **Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events"**, *Journal of
   Financial Econometrics*; arXiv:1011.6402.
   Shows order flow imbalance (OFI) at the best quotes has an approximately linear relationship
   with short-horizon price changes, robust across 50 NYSE stocks. This is the direct ancestor
   of this project's $M_0$ baseline feature ($\text{OFI}_{L1}$) — the project's finding that
   $M_0$ alone already achieves PR-AUC ≈0.63 (§14.1) is consistent with OFI's well-documented
   predictive power here.

8. **Huang, Lehalle & Rosenbaum (2015), "Simulating and Analyzing Order Book Data: The
   Queue-Reactive Model"**, *Journal of the American Statistical Association*; arXiv:1312.0563.
   Models the LOB as a Markov queuing system whose arrival/cancellation/market-order intensities
   are state-dependent (a function of current queue sizes). This is the closest prior framework
   to the project's Target B (pure depth depletion) — where this paper treats queue evolution as
   Markovian in queue *state*, the project's $M_2$ instead treats depletion-relevant intensities
   as Hawkes-history-dependent (self-exciting in *event history*, not just current depth); the
   two are complementary rather than competing formalizations of the same phenomenon.

9. **Gould & Bonart (2016), "Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit
   Order Book"**, *Quantitative Finance* / arXiv:1512.03492.
   Finds L1 bid/ask queue imbalance is a strong, statistically significant, near-linear predictor
   of the next mid-price tick move, especially for large-tick instruments. Directly supports this
   project's inclusion of absolute queue sizes and rolling percentile rank in $M_0$ (PROJECT_SPEC
   §2.4 explicitly calls this "critical — omitting this makes $M_0$ artificially weak").

10. **Moallemi & Yuan (2016/2017), "A Model for Queue Position Valuation in a Limit Order
    Book"**, Columbia Business School Research Paper No. 17-70.
    *Flag: found only as a working paper (SSRN / Columbia working-paper series); no confirmed
    peer-reviewed journal publication was found — cite as a working paper, not a journal article.*
    Models the economic value of queue position combining informational and queueing effects,
    with cancellation intensity increasing in queue size — a direct antecedent of this project's
    Target B (pure depth depletion via queue cancellation dynamics) and its `qty_down` /
    cancellation-volume event types (§3.1 of PROJECT_SPEC.md).

## (c) Machine learning (gradient boosting, deep learning, transformers) for LOB forecasting

11. **Kercheval & Zhang (2015), "Modelling high-frequency limit order book dynamics with support
    vector machines"**, *Quantitative Finance* 15(8):1315–1329.
    Multi-class SVM on LOB level attributes to predict mid-price movement direction. One of the
    earliest ML-for-LOB papers and the natural precedent that this project's $M_1$ (LightGBM on
    an engineered feature bank) updates with a modern nonlinear tree ensemble in place of an SVM.

12. **Ntakaris, Magris, Kanniainen, Gabbouj & Iosifidis (2018), "Benchmark dataset for mid-price
    forecasting of limit order book data with machine learning methods"**, *Journal of
    Forecasting* 37(8); arXiv:1705.03233. (The FI-2010 dataset paper.)
    Establishes the FI-2010 benchmark (Nasdaq Nordic, 10 days, 5 stocks, ~4M samples,
    3-class up/stationary/down labels at multiple horizons) that a large share of the ML-for-LOB
    literature below trains and evaluates on. Useful context for how far this project's own
    labeling protocol (Target A/B, adverse-move / depth-depletion classification, §2.1 of
    PROJECT_SPEC.md) departs from the field's dominant 3-class mid-price-direction convention —
    this project instead predicts depletion/adverse-move events specifically, not price direction.

13. **Zhang, Zohren & Roberts (2019), "DeepLOB: Deep Convolutional Neural Networks for Limit
    Order Books"**, *IEEE Transactions on Signal Processing* 67(11):3001–3012; arXiv:1808.03668.
    CNN+LSTM architecture that was state-of-the-art on FI-2010 at publication and generalized
    (out-of-sample, across instruments) on LSE data. Represents the deep-learning end of the
    $M_1$-style "unconstrained representation learning" side of this project's central contrast,
    even though this project itself uses LightGBM rather than a CNN/LSTM as its free-model
    baseline — worth citing as the reason a gradient-boosted tree (rather than a deep net) is a
    defensible, cheaper stand-in for "unconstrained feature learning" at this project's much finer
    L1/event-level (rather than book-snapshot-image) granularity.

14. **Sirignano & Cont (2019), "Universal features of price formation in financial markets:
    perspectives from deep learning"**, *Quantitative Finance* 19(9); arXiv:1803.06917.
    A pooled, multi-stock LSTM outperforms asset-specific linear/nonlinear models and generalizes
    to held-out stocks, evidence for a "universal" price-formation mechanism learnable from data
    without instrument-specific structure. This is the most direct precedent — for a *different*
    axis of transfer (across-instrument pooling of a free model) — against which this project's
    Core Experiment 2 (transferring the *parametric* Γ matrix, not pooling raw ML training data,
    across BTC→ETH→SOL→DOGE) can be contrasted: Sirignano & Cont pool data into one free model,
    this project instead transfers a compressed parametric object and re-profiles only μ.

15. **Wallbridge (2020), "Transformers for Limit Order Books"**, arXiv:2003.00130 (TransLOB).
    *Flag: found as an arXiv preprint; a confirmed peer-reviewed publication venue was not
    located — cite as a preprint.*
    CNN-feature-extraction-plus-Transformer architecture for LOB mid-price prediction on
    FI-2010, representative of the most recent (attention-based) rung of the "free model"
    ladder this project's $M_1$ sits below in complexity.

## (d) Papers directly comparing a parametric point-process model against an ML model

**This is the category where the literature search struggled most, and that gap is itself worth
reporting rather than papering over with a loose fit.** No paper was found that runs the same
head-to-head this project runs: a constrained, MLE-fit multivariate Hawkes process against an
unconstrained tree ensemble, on identical LOB targets, under a formal statistical-significance
protocol (Giacomini-White + Romano-Wolf). The closest matches found:

16. **Shi & Cartlidge (2022), "State Dependent Parallel Neural Hawkes Process for Limit Order
    Book Event Stream Prediction and Simulation"**, KDD '22 (28th ACM SIGKDD Conference on
    Knowledge Discovery and Data Mining).
    Builds a *hybrid* neural-Hawkes model (continuous-time LSTMs modeling per-event-type
    intensities, plus an event-state interaction mechanism) and reports it outperforms both pure
    stochastic (Hawkes) and pure deep-learning baselines on LOB event-stream prediction. This is
    the closest thing found to a controlled Hawkes-vs-ML comparison in the LOB literature, but it
    is a hybridization/fusion study (arguing combination beats either alone), not a controlled
    "does the constrained parametric model generalize better under small-N / transfer" comparison
    of the kind this project runs — the research question is adjacent, not equivalent.

17. **Nittur Anantha & Jain (2026), "Forecasting High Frequency Order Flow Imbalance using
    Hawkes Processes"**, *Computational Economics* 67(1):279–312; arXiv:2408.03594.
    Reports that a sum-of-exponentials Hawkes process "gives the best forecast among all
    competing models" for near-term OFI distribution forecasting on NSE tick data. *Flag: the
    publicly available abstract/summary does not name the specific competing models (whether they
    include tree ensembles, LSTMs, or only classical time-series baselines like ARIMA); this could
    not be confirmed without full-text access, so do not describe this paper as confirmed to
    include a gradient-boosting or deep-learning baseline without checking the full text first.*
    If its competing-model set does include ML baselines, it would be the single closest
    published match to this project's exact research question (Hawkes vs. ML on order-flow
    prediction) — but that has not been verified here, and should not be claimed as confirmed
    in the paper without checking the full text.

18. **Mei & Eisner (2017), "The Neural Hawkes Process: A Neurally Self-Modulating Multivariate
    Point Process"**, NeurIPS 2017; arXiv:1612.09328.
    Not finance-specific, but the foundational paper for the "replace/augment the Hawkes kernel
    with a neural network" line of work that papers #16 above build on. Relevant background for
    framing why this project deliberately keeps $M_2$ *fully parametric* (exponential kernels,
    closed-form MLE, explicit Γ) rather than following this line toward a neuralized intensity —
    the whole point of $M_2$ in this project is the falsification-first test of whether the
    parametric *constraint itself* is what helps, which a neural Hawkes model would no longer
    isolate.

**Honest summary of the gap:** the field has (i) many papers on parametric Hawkes for order flow,
(ii) many papers on ML/deep learning for LOB prediction, and (iii) a small number of *hybrid*
papers that fuse the two architectures and show the fusion wins. What is largely absent is a
paper that holds the *prediction target and the tree/ML architecture fixed* and asks, as this
project does, whether a Hawkes-constrained feature representation is a better regularizer than
letting a tree see the raw EWMA feature bank directly — i.e., an ablation-style comparison rather
than an architecture-fusion or bake-off study. This project's Core Experiment 1 (learning-curve
crossover search) appears to be a genuinely novel angle on the Hawkes-vs-ML question relative to
the papers found here.

## (e) Statistical methodology: predictive-ability testing and multiple-testing correction

19. **Giacomini & White (2006), "Tests of Conditional Predictive Ability"**, *Econometrica*
    74(6):1545–1578.
    Introduces the Giacomini-White (GW) test framework this project uses directly (§5.1 of
    PROJECT_SPEC.md): out-of-sample predictive-ability testing valid even when forecasting models
    are estimated/re-estimated on a rolling basis and potentially misspecified, extending
    Diebold-Mariano/West to a genuinely walk-forward comparison. This project's minute-averaged,
    HAC-standard-error, rolling-reestimation GW protocol (STATUS_M3.md, §14.2 of
    PROJECT_SPEC.md — GW statistic ≈ −38.6, p≈0) is a textbook application of this paper's method.

20. **Romano & Wolf (2005), "Stepwise Multiple Testing as Formalized Data Snooping"**,
    *Econometrica* 73(4):1237–1282.
    The original stepdown methodology for controlling the family-wise error rate (FWER) across
    a family of related hypothesis tests via resampling, explicitly framed around the
    data-snooping problem in empirical finance (multiple strategies/specifications tested on the
    same data). Direct methodological ancestor of §5.2's Romano-Wolf stepdown across the
    secondary 18-specification grid, resampled by day-blocks to preserve intraday dependence.

21. **Romano & Wolf (2005), "Exact and Approximate Stepdown Methods for Multiple Hypothesis
    Testing"**, *Journal of the American Statistical Association* 100(469):94–108.
    A companion methodological paper (same authors, same year) developing the exact/approximate
    stepdown procedures underlying the Romano-Wolf method; commonly cited alongside the
    Econometrica paper above as the joint methodological basis for Romano-Wolf FWER control.
    Cite both together, as is standard practice, rather than either alone.

---

## Summary counts

| Category | Verified papers |
|---|---|
| (a) Hawkes / point processes in finance | 6 |
| (b) LOB modeling / queue-position & depletion prediction | 4 |
| (c) ML (GBM / deep learning / transformers) for LOB forecasting | 5 |
| (d) Direct Hawkes-vs-ML comparison | 3 (none is a clean, exact match — see honest gap note above) |
| (e) Statistical methodology (GW test, Romano-Wolf) | 3 |
| **Total** | **21** |

No fabricated citations are included. Three entries carry explicit flags on venue/peer-review
status that could not be fully confirmed (Hewlett 2006 — workshop paper, not a journal article;
Moallemi & Yuan — working paper, no confirmed journal publication found; Wallbridge 2020 —
arXiv preprint, no confirmed peer-reviewed venue found). These are still real, findable, citable
works — the flag is about publication status, not about whether the work exists.
