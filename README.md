# Spectral Market Connectivity

Python research on market co-movement, graph spectra and the limits of connectivity forecasts.

Can the structure of a market's correlation network tell us more than a simple average correlation? This project builds reproducible price panels, studies their positive and negative relationships, and tests whether past graph measurements help predict future connectivity.

**Current conclusion:** the graph diagnostics reveal useful descriptive structure, but the forecasting experiments have not established a dependable advantage. This is ongoing research, not a trading system or an arbitrage result.

![Early ETF return relationships](output/market_connectivity/figures/R00_early_correlation_wide.png)

Rows and columns identify the fixed funds. Color shows Pearson correlation of the first 1,000 common daily returns, February 2006–January 2010. The stock-sector block differs from Treasury, gold and commodity-fund exposures. This describes that period, not permanent hedges or future price direction.

## Run the offline demo

Use Python 3.13.6, the validated interpreter. Create a virtual environment, activate it using your operating system's usual command, then run:

```bash
python -m pip install -r requirements.txt
python synthetic_demo.py
python run_checks.py
```

The seeded demo needs no account, API key, internet connection or market-data archive. It creates a **clearly labeled synthetic** correlation heatmap, rolling diagnostic figure and summary under `demo_output/`. The checks test graph identities, missing-data handling, cache integrity and forecast chronology on synthetic inputs. They do not download market data or fit a new market forecast.

The default requirements use the exact tested core versions in `requirements-validated.txt`. These pin direct dependencies rather than every transitive package. Newer NumPy/Pandas/SciPy versions failed a synthetic trajectory check during the clean-environment audit, so broader compatibility is not claimed. [Validation metadata](release_validation.json) records the exact environment and commands used for this export.

## What is included

- **Graph mathematics:** positive, negative, signed and shifted operators; eigenvalues, modes, normalization, localization and common-sample diagnostics.
- **Data engineering:** explicit requests, immutable gzip/JSON snapshots, checksums, provenance, missingness/identity rules and a local SQLite catalog.
- **Research:** original equity reproduction, cross-market screens, fixed-cohort crypto studies and chronological ETF connectivity forecasts.
- **Evidence:** the frozen ETF protocol, all 250 aggregate model-comparison rows from 50 cases, selected Python figures and concise explanations.

All active scientific calculations and plots use Python. Presentation exports display completed calculations. Raw provider observations, local archives, deployment material and credentials are intentionally absent. Some archive-dependent modules are included for inspection and reproduction with authorized local data; they are not part of the archive-free quickstart.

## A measured result, including the negative evidence

The fixed ETF cohort is XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, GLD, TLT and DBC. Its local frozen source supplies 5,200 complete daily returns from February 2006 through October 8, 2026. The cohort was selected retrospectively using coverage and early descriptive diversity, not forecast scores.

For the primary 60-return graph and 20-session future-average target, the models share **1,241 historical evaluation dates**:

| Dimensionless graph target | No-change MAE | Current-state model MAE | Added-history model MAE |
| --- | ---: | ---: | ---: |
| Positive Laplacian second eigenvalue / 12 | 0.018506 | 0.015966 | 0.016729 |
| Mean negative pair weight | 0.006672 | 0.006227 | 0.006159 |

The positive history model has higher observed error than the current-state model. The negative history model's improvement is small and uncertain. Paired block intervals include no improvement in both primary comparisons. Nominal 90% intervals cover 83.2% and 89.0% of outcomes. These results do not establish trustworthy future probabilities or an executable financial edge.

![Positive connectivity forecast errors](output/market_connectivity/figures/R04_positive_forecast_error.png)

The horizontal axis is mean absolute error in dimensionless score units, so smaller is better. The vertical axis lists models evaluated on identical dates. Predicting a graph score is different from predicting whether prices rise or fall.

Read [findings](docs/FINDINGS.md), [methods](docs/METHODS.md), [data availability](docs/DATA_AVAILABILITY.md) and the [source map](docs/SOURCE_MAP.md). [Figure guide](docs/FIGURES.md) explains every included graph's axes. The complete aggregate comparison is [CSV](output/market_connectivity/comparison.csv).

## Reproduction and limitations

The offline demo and checks reproduce algorithms without third-party data. Exact real-data reproduction additionally requires the original lawfully obtained snapshots with the recorded hashes. A new provider query may return revised data and is a new snapshot, not exact reproduction. [Data instructions](docs/DATA_AVAILABILITY.md) explain the expected paths and optional acquisition dependencies.

Overlapping windows are dependent. Today's fixed asset cohort is not a point-in-time historical universe. Adjustment histories, release timing, changing composition, finite-sample correlations and provider revisions remain material limitations. Wholesale power prices, retail state averages and physical grid operations are different datasets. Market-price correlation is not a map of transmission capacity or a causal mechanism.

Project status: ongoing exploratory research. Next work should freeze a genuinely prospective evaluation and test whether graph features improve a concrete risk target beyond strong simpler baselines.

Repository: [Siroxpee/spectral-market-connectivity](https://github.com/Siroxpee/spectral-market-connectivity).
