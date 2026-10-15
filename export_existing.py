"""Export your existing DAILY notebook prices, without a network download."""
from pathlib import Path
import pandas as pd
from data_cache import cached_closes, validate_prices


def export_current_prices(close_prices, output_dir="current_price_snapshot"):
    """Save the in-memory matrix while labeling unknown original settings honestly.

    Call this in the original notebook kernel, where close_prices exists.
    Bounds describe the supplied snapshot, not a recovered original API query.
    Return the folder to zip and attach. No credentials are written.
    """
    validate_prices(close_prices)
    if not close_prices.index.equals(close_prices.index.normalize()):
        raise ValueError("This export helper is for your daily notebook snapshot.")
    request = {
        "source": "existing yfinance notebook Close matrix",
        "symbols": close_prices.columns.tolist(),
        "start": close_prices.index[0].isoformat(),
        "end": (close_prices.index[-1] + pd.Timedelta(days=1)).isoformat(),
        "interval": "1d",
        "adjustment": "UNVERIFIED: original auto_adjust not explicitly recorded",
        "feed": "Yahoo public interface; original retrieval time not recorded",
        "timezone_policy": "preserve original daily labels",
        "universe_as_of": "UNVERIFIED: original scrape date not recorded",
    }
    stored_prices, manifest, hit = cached_closes(request, lambda: close_prices, output_dir)
    if hit:
        try:
            # CSV preserves timestamps but its parser may choose a different
            # storage unit (seconds versus microseconds in pandas 3). Compare
            # exact instants at the input unit, without allowing truncation.
            comparison_prices = stored_prices.copy(deep=False)
            comparison_prices.index = stored_prices.index.as_unit(
                close_prices.index.unit, round_ok=False
            )
            pd.testing.assert_frame_equal(
                close_prices, comparison_prices, check_exact=True, check_freq=False
            )
        except (AssertionError, ValueError) as error:
            raise ValueError(
                "The existing frozen snapshot differs from the supplied daily prices "
                "(values, labels, dtypes, or index metadata). Export to a new output_dir "
                "to preserve both snapshots."
            ) from error
    print(f"Saved/reused {manifest['rows']} price rows and {len(manifest['actual_symbols'])} symbols; cache_hit={hit}")
    print("Original adjustment, retrieval time and package versions remain unverified.")
    return str(Path(output_dir).resolve())
