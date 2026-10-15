# Spectral Market Connectivity

**An ongoing stock-market research project by Mathew Tkachev.**

Can the relationships between stocks help describe market structure and, eventually, changing market risk?

This study starts with daily stock prices, measures how stocks move together, turns those relationships into a weighted graph, and studies its Laplacian eigenvalues and eigenvectors. The current baseline uses 503 stock labels over 2016–2026 and 2,643 trailing 60-observation windows.

**Current finding:** the second eigenvalue often closely follows the weakest-connected stock. In 84.4% of windows it lies within 1% of a mathematical bound determined by that stock's total connection weight. This helps explain what the measurement captures. It does **not yet establish a forecast of market risk**. Source-quality review and industry-level interpretation remain unfinished.

## Read the study

Start with **[RESEARCH.md](RESEARCH.md)**. It contains the complete working paper: questions, data provenance, cleaning pathway, equations, figures with axis explanations, findings, limitations, and next steps. Figures appear inside the paper; their image files live in `figures/`.

## Repository map

| File or folder | Purpose |
|---|---|
| [RESEARCH.md](RESEARCH.md) | One detailed research document, including unfinished sections. |
| [spectral_core.py](spectral_core.py) | Common cleaning, correlation, Laplacian, eigenvalue and localization calculations. |
| [stock_data.py](stock_data.py) | Read and validate a local stock-price panel; calculate returns without filling gaps. |
| [stock_analysis.py](stock_analysis.py) | Run the stock study and create labeled scientific figures. |
| [checks.py](checks.py) | Compact mathematical and data-cleaning checks. |
| [data/dataset.json](data/dataset.json) | The stock dataset's dimensions, provenance limits, settings and file fingerprints. |
| [results/window_metrics.csv](results/window_metrics.csv) | The 2,643 saved window-level measurements used by the paper. |
| [results/findings.json](results/findings.json) | Numerical findings, selected-window diagnostics and exploratory controls. |
| `figures/` | Seven Python-generated images displayed in the paper. |
| [requirements.txt](requirements.txt) | Tested Python package versions. |
| `.gitignore`, `.gitattributes` | Keep local caches/data out of commits and preserve predictable text/image handling. |

The original recovered price snapshot remains local; this repository includes its description and derived results. The original adjustment settings and historical constituent membership are unverified. A fresh download creates a new snapshot or version of the panel, not an exact replacement for the original snapshot.

Python 3.13.6 was used for this edition. To check the mathematics: `python checks.py`. See `python stock_analysis.py --help` for the explicit analysis commands. No data acquisition runs when these modules are imported.
