# Methods and information timing

## Source to graph

1. Declare the complete panel: source, ordered universe, interval, span, adjustment, quote currency, units and timestamp convention. A ticker or rolling window is not a separate dataset.
2. Preserve native source evidence and a manifest with hashes. Validate ordering, duplicates, finite values and identity assumptions. Hash agreement checks bytes, not independent market truth.
3. Positive financial levels use arithmetic returns `r[t] = P[t]/P[t-1] - 1` with no implicit filling. Signed physical price series use first differences `P[t]-P[t-1]` in their source units. Source-specific transformations are explicit.
4. Preserve source calendars. Continuous intraday inputs mask gap-spanning changes according to the declared bar length. US ETF Friday-to-Monday observations remain adjacent trading sessions. Missing observations never become zero or forward-filled prices.
5. The original baseline uses 60 observations, removes any asset incomplete within that window, and removes sample standard deviation at or below 1e-12. Later fixed-cohort studies instead mark an incomplete common window invalid; they do not silently shrink the fixed universe. All graph diagnostics use the same clean window.
6. Compute Pearson `C`. The positive graph has `W+ = max(C,0)` off diagonal and zero diagonal. The degree matrix is `D = diag(W+ 1)`, and `L+ = D-W+`. Its second-smallest eigenvalue is algebraic connectivity. Disconnected support gives zero.

Negative-magnitude edges use `W- = max(-C,0)`, separately. Their mean averages all unordered pairs, including zero weights. Signed `Ls = diag(sum(abs(S)))-S` has different lowest-mode semantics and is never relabeled as ordinary algebraic connectivity. Dividing lambda2 by node count is not the same operator as the degree-normalized Laplacian `I-D^(-1/2)WD^(-1/2)`.

Raw lambda2 depends on graph size, edge scale, composition and weak nodes. A high or low value does not directly mean high or low financial risk. Correlations measure co-movement, not price gains, causes, probabilities or physical transfer capability.

## Forecast experiment

The [ETF protocol](../output/market_connectivity/protocol.json) preserves exact target and chronology definitions. W = 20/40/60/120 observations; K = 20/60/100 past graph-score endpoints; H = 20/60/100 sessions. The main positive score is lambda2(L+)/12; the opposing score is mean W- over all 66 unordered pairs.

Future-average and endpoint targets are separate: `mean(s[t+1:t+H+1])-s[t]` versus `s[t+H]-s[t]`. A future-average target is not a new graph built solely from future raw returns. At W60/H20, future rolling windows contain 82.5% already-known return slots on average. K100 with W60 spans 159 raw returns. Neither count implies independent observations or explained variance.

Expanding chronological forecast blocks train only on labels completed strictly before their origins. Earlier inner data tune ridge strength and K. Calibration errors have their own earlier forecast origins. Training-only feature imputation/standardization occurs after source cleaning; model-feature medians never fill source prices. Comparisons use identical target dates and strong persistence, mean-reversion and current-state baselines.

Paired circular-block uncertainty is conditional on saved forecasts. It is not a correction for 50 examined cases or a guarantee under regime drift. The entire study is retrospective and exploratory. An October 8 archived-as-of projection generated October 15 was already partly elapsed at issuance and is not a live forecast.

## Verification

The offline checks cover graph identities, PSD/trace and signed semantics, missing/constant data, exact cache round trips, schema/hash corruption, future perturbation guards, strict horizon purges and train-only transforms. Source parsers have separate mocked checks for DST, failed providers, missing assets and invalid raw records. Passing these checks verifies implementation properties, not predictive usefulness.

Cache publication includes a bounded retry for Windows directory-rename access/sharing errors (5, 32 and 33). It checks for destination conflicts on every attempt and never overwrites an existing snapshot. Other errors propagate. This storage portability adjustment does not change graph definitions, models or frozen observations.

Primary methodological context: [von Luxburg, spectral clustering](https://arxiv.org/abs/0711.0189), [Diebold and Mariano, dependent forecast comparison](https://doi.org/10.1080/07350015.1995.10524599), [Gneiting and Raftery, proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
