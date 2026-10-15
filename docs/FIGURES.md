# Figure reading guide

Each figure is a Python-generated analytical derivative. No raw per-asset price feed is distributed. Network/workflow layout positions have no quantitative measurement unless an axis explicitly says otherwise.

## Different exposures produce different links

![Figure](../output/market_connectivity/figures/R00_early_correlation_wide.png)

Rows/columns identify funds. Color is return correlation: green together, brown opposite. This early check does not prove predictability.

## The structure changes through time

![Figure](../output/market_connectivity/figures/R03_score_history.png)

x: observed session date. y: each dimensionless score, on its own scale. These describe relationships—not whether prices rose or fell.

## Three lengths answer different questions

![Figure](../output/market_connectivity/figures/R01_window_timing.png)

x: return sessions relative to today. Rows show observation spans. W60 × K100 uses 159 returns; H20 adds 20 future sessions.

## The complete data-to-forecast pathway

![Figure](../output/market_connectivity/figures/R02_cleaning_pathway.png)

Read rows left to right, then down. Positions are not measured axes. No prices or returns were filled.

## Positive connectivity: history is not the best model

![Figure](../output/market_connectivity/figures/R04_positive_forecast_error.png)

x: average error; smaller is better. y: forecast method. Added history reduced error 9.6% versus no change; extra value beyond current state is uncertain.

## Opposing moves: a modest measured improvement

![Figure](../output/market_connectivity/figures/R05_negative_forecast_error.png)

x: average error; smaller is better. y: forecast method. Added history reduced error 7.7% versus no change; extra value beyond current state is uncertain.

## More history does not help every setting

![Figure](../output/market_connectivity/figures/R06_window_horizon_grid.png)

Rows: graph length W. Columns: horizon H. Color/number: error reduction versus current state. Positive helps; negative hurts. All cells are exploratory.

## A curve describes the past, not a guarantee

![Figure](../output/market_connectivity/figures/R08_empirical_distributions.png)

x: observed W60 score. y: fraction of windows at or below it. Shifting curves show changing distributions; jumps at zero mark disconnected graphs.

## Probability estimates still need caution

![Figure](../output/market_connectivity/figures/R07_probability_calibration.png)

x: estimated chance. y: observed frequency. Closer to diagonal is better. Nominal 90% bands covered 83.2% positive / 89.0% opposing outcomes.

## Future average and endpoint are separate tests

![Figure](../output/market_connectivity/figures/R09_average_vs_endpoint.png)

x: forecast methods. y: error reduction versus no change. Green averages the next 20 scores; brown predicts only the final score.
