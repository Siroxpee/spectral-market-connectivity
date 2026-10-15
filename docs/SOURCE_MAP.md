# Source map

All root Python files copied into this release retain their source bytes. The release adds only `synthetic_demo.py` and `run_checks.py` as standalone public entry points. `RELEASE_MANIFEST.json` records file hashes.

| Area | Entry points | Data requirement |
| --- | --- | --- |
| Offline demonstration | `synthetic_demo.py`, `run_checks.py` | None; seeded synthetic arrays |
| Core graph mathematics | `spectral_core.py`, `crypto_core.py`, `checks.py` | DataFrame/array inputs supplied by caller |
| Storage and provenance | `data_cache.py`, `archive_catalog.py`, `export_existing.py` | Local synthetic or authorized native observations |
| Provider adapters | `market_sources.py`, `collect_data.py`, `financial_expansion.py`, `crypto_acquisition.py`, `electricity_acquisition.py`, `retail_power_acquisition.py` | Explicit network operations; optional dependencies/provider terms |
| EIA grid preparation | `download_eia_grid.py`, `prepare_eia_grid.py` | Official source ZIPs; hour-ending convention |
| General offline research | `expanded_spectral.py`, `organize_data.py`, `reproduce_daily.py` | Frozen snapshots and matching inventory/catalog |
| Fixed ETF experiment | `market_connectivity_data.py`, `market_connectivity.py` | Exact frozen source for full run; self-test needs only included protocol |
| Fixed crypto experiments | `crypto_mvp_data.py`, `crypto_mvp_research.py`, `crypto_trajectory_data.py`, `crypto_trajectory.py` | Original authorized archives for full run; self-tests are synthetic |
| Python reports/figures | `market_connectivity_report.py`, `report_charts_*.py`, `build_findings_report.py`, `build_expansion_dashboard.py`, `draw_expansion_workflow.py` | Completed local results and relevant archive |
| Presentation-only exports | `export_dashboard_data.py`, `export_python_dashboard.py` | Local preview/source metadata; scientific results generated in Python |
| Full-archive verification | `verify_expansion.py`, `verify_python_pipeline.py`, `sync_expansion.py` | Full corresponding local assets; not the compact quickstart |
| Adapter and integrity tests | `check_*.py` | Synthetic/mock inputs; optional parser dependencies for adapter checks |

The source expects paths relative to the repository root. Full archive workflows are intentionally preserved for inspection rather than replaced by undocumented provider lookups. Original submitted notebooks, deployment identities, private previews, PDF rendering tied to workstation fonts and one-time metadata-repair scripts are excluded.
