# Data availability and reproduction

## Public package versus local research archive

This release contains Python source, synthetic examples, frozen experiment definitions, aggregate performance metrics and research figures. It contains **no raw per-asset/provider price or return feed**, API response archive, provider workbook, credentials or local SQLite database. The private research archive is roughly 3 GB and is not required for the offline demo.

Third-party access, retention and redistribution rights are separate from the project code. Obtain data through the provider under terms applicable to your own use. Public accessibility is not a blanket redistribution license. This release makes no promise of unrestricted historical retention, commercial use, current prices or exact reproduction from a new network query. No acquisition command runs during installation or the quickstart.

## Offline checks without market data

`python synthetic_demo.py` and `python run_checks.py` use the tested direct core pins selected by `requirements.txt`, with Python 3.13.6. Broad compatibility with newer numerical-library versions is not claimed. For mocked source-adapter tests, install `requirements-acquisition.txt`, then run:

```bash
python check_sources.py
python check_financial_expansion.py
python check_electricity_acquisition.py
python check_crypto_acquisition.py
```

These checks use local synthetic/mocked responses, not market downloads. Legacy XLS acquisition additionally requires xlrd; XLSX uses openpyxl. Timezone data are explicit dependencies for Windows compatibility. Acquisition support is optional and provider-specific.

## Exact real-data rerun

The frozen ETF descriptor is `output/market_selection/selected_panel.json`. Exact reproduction expects:

```text
data/expanded/financial/snapshots/
  6803319d2d24e5a4ff01ecb483f1bd1872e0c69972cbbbfa1d5a69c47b75a7a5/
    manifest.json
    prices.csv.gz
```

The gzip SHA-256 is `d279458ba10b2cb9e30986b073055c2d1912d85eda5e7b42aa9e1331153de235`; prepared fixed-12 return-matrix SHA-256 is `27bc129e471cfa83dfc24361591752844d384dc008cf66a0a08f80c3fc4f83be`. The source is Yahoo/yfinance `auto_adjust=True`. Provider session dates have observed SPY support but no independently reconciled full exchange calendar or verified publication timestamps. Historical adjustment/availability was not reconstructed point in time.

If you already have authorized copies of those exact files, preserve the relative paths and run `python market_connectivity.py`. This computes all 50 cases and may take substantial time. The public release includes the protocol and aggregate comparison but omits dated predictions/checkpoint arrays. Rebuilding the full report requires the new local experiment outputs before `python market_connectivity_report.py`.

If you do not have the original snapshot, use the demo or declare a new provider request and a new protocol/source version. `market_sources.py` and `collect_data.py` show explicit adapter and snapshot APIs. Financial, electricity and crypto acquisition CLIs can create network traffic and write local archives; inspect their arguments and source terms first. Do not replace a frozen source under the same hash or reuse its scores as though the inputs were unchanged.

The original-stock reproduction needs its own snapshot/catalog and original reference outputs. General expansion reports and archive validators need the full corresponding inventory and local files. They are source-inspection/full-archive utilities, not commands promised to work from this compact release alone.

## Source families and interpretation

| Source | Local research role | Main limitation |
| --- | --- | --- |
| [Yahoo/yfinance](https://ranaroussi.github.io/yfinance/) | Stocks/ETFs, indicative FX, nearby futures proxies | Unofficial interface, revisions/adjustment, limited recent intraday access, incomplete historical membership and unidentified futures rolls |
| [ECB reference rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html) | Daily currency-reference quotes | EUR numeraire and reference fixing, not an executable synchronized FX book |
| [EIA](https://www.eia.gov/electricity/wholesale/) / ISO sources | Wholesale gas/power, retail electricity, grid operations | Different products, reporting units, release timing, sparse quotes, revisions and DST/hour-ending semantics |
| [World Bank commodity prices](https://www.worldbank.org/en/research/commodity-markets) | Monthly physical price assessments | Product/contract definitions, mixed units, aggregate/component overlap, reference months rather than releases |
| [Binance public data](https://github.com/binance/binance-public-data) | Fixed-cohort crypto source examples and local experiments | USDT quote, single venue, survivor/identity limitations; [dataset terms](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) govern data and derivatives |

No Binance-derived research arrays or figures are included in this release. Its Python implementation and synthetic self-tests are included. Source checksums verify file bytes, not independent price accuracy. A new request can fail or differ because providers change their interfaces and histories.

EIA retail monthly values are sales-weighted cents/kWh, not wholesale LMP. ISO market locations are not physical transmission lines. EIA grid source timestamps are hour ending even where a legacy CSV index name says interval start. Align delivery conventions before joining data. Converting one unit by a fixed positive scale leaves Pearson correlation unchanged, but time-varying currency conversion or mismatched aggregation can create misleading links.
