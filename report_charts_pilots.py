"""Render existing collection pilots and a research workflow diagram, offline.

No provider requests run here. The three fixed catalog keys are the frozen
ETF, ECB and Coinbase panels already collected for this project. Indexed
levels show descriptive movements; they do not fit or test a forecast.
"""

from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from archive_catalog import load_snapshot


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "data" / "archive_catalog.sqlite"
OUTPUT = ROOT / "output" / "findings"
FIGURES = OUTPUT / "figures"
NAVY, TEAL, ORANGE = "#17324D", "#007F86", "#D57B27"
COLORS = [NAVY, TEAL, ORANGE, "#76558F", "#56824A", "#4779A8", "#B24C6B", "#7C6A53"]
KEYS = {
    "etf": "5dbd58dbc0b4517fa5fb14cf5206d367cb2e64d37794d7c47905996618bb612f",
    "fx": "9617257f3fbb7314a9815ba9d5165566ac0941b342e1c5fc3bdf4bec95bb33ea",
    "crypto": "f5ca2d3f6d0ef642531b961c0ade7472ab864fc17dd9df99f238e12f03b4518b",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelcolor": NAVY, "text.color": NAVY,
    "axes.edgecolor": "#CCD6DE", "xtick.color": NAVY, "ytick.color": NAVY,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
})


def panel(name):
    """Read exactly one existing registered cache, with integrity checks."""
    prices, manifest = load_snapshot(KEYS[name], CATALOG)
    # The locations are part of this catalog, not a search for replacement data.
    from archive_catalog import list_snapshots
    entry = next(row for row in list_snapshots(CATALOG) if row["snapshot_key"] == KEYS[name])
    folder = Path(entry["snapshot_path"])
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    if provenance["price_file_sha256"] != manifest["sha256"]:
        raise ValueError("Pilot source evidence does not match the frozen price file.")
    return prices, manifest, provenance, folder


def indexed(prices):
    """Divide by each asset's first actual observation, without filling gaps."""
    first = prices.apply(lambda column: column.loc[column.first_valid_index()])
    return 100.0 * prices.div(first, axis="columns")


def stats(prices, manifest):
    endpoint = 100.0 * (prices.iloc[-1] / prices.iloc[0] - 1.0)
    return {
        "rows": len(prices), "symbols": prices.columns.tolist(),
        "symbol_count": len(prices.columns),
        "observed_cells": int(prices.notna().sum().sum()),
        "missing_cells": int(prices.isna().sum().sum()),
        "observations_by_symbol": {symbol: int(prices[symbol].notna().sum()) for symbol in prices},
        "actual_first_label": prices.index[0].isoformat(),
        "actual_last_label": prices.index[-1].isoformat(),
        "first_to_last_price_or_quote_change_pct": {symbol: float(endpoint[symbol]) for symbol in prices},
        "snapshot_price_sha256": manifest["sha256"], "request": manifest["request"],
    }


def sources(folder, provenance):
    return [str(folder / "manifest.json"), str(folder / "provenance.json"), provenance["source_url"]]


def style_axis(axis):
    axis.grid(axis="y", color="#E4EAF0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.axhline(100, color="#A7B6C5", linestyle="--", linewidth=0.8)


def save(figure, filename):
    axes = [{"x": ax.get_xlabel(), "y": ax.get_ylabel(), "x_visible": ax.xaxis.get_visible(),
             "y_visible": ax.yaxis.get_visible(), "kind": "measurement"} for ax in figure.axes]
    for item in axes:
        if (item["x_visible"] and not item["x"]) or (item["y_visible"] and not item["y"]):
            raise ValueError(f"Missing measurement axis label in {filename}: {item}")
    (FIGURES / (Path(filename).stem + ".axes.json")).write_text(json.dumps(axes, indent=2), encoding="utf-8")
    figure.savefig(FIGURES / filename, dpi=180, facecolor="white")
    plt.close(figure)


def etf_chart():
    prices, manifest, provenance, folder = panel("etf")
    local = prices.index.tz_convert("America/New_York")
    levels = indexed(prices)
    sessions = sorted(set(local.date))
    if len(sessions) != 2:
        raise ValueError("G16 expects the already collected two-session ETF pilot.")
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    figure.subplots_adjust(left=0.09, right=0.975, bottom=0.25, top=0.76, wspace=0.18)
    figure.text(0.09, 0.935, "G16  |  Equity ETF collection pilot", fontsize=19, weight="bold")
    figure.text(0.09, 0.87, "SPY, QQQ and IWM • October 1–2, 2026 • 5-minute closes", fontsize=11)
    for axis, session in zip(axes, sessions):
        mask = local.date == session
        clock_minutes = local[mask].hour * 60 + local[mask].minute
        for color, symbol in zip(COLORS, prices.columns):
            axis.plot(clock_minutes, levels.loc[mask, symbol], color=color, linewidth=1.8, label=symbol)
        axis.set_title(f"October {session.day}  ·  {int(mask.sum())} bars", fontsize=12, pad=12)
        axis.set_xlim(570, 955)
        axis.set_xticks([570, 660, 750, 840, 955], ["09:30", "11:00", "12:30", "14:00", "15:55"])
        axis.get_xticklabels()[0].set_horizontalalignment("left")
        axis.get_xticklabels()[-1].set_horizontalalignment("right")
        axis.set_xlabel("New York clock time (EDT; hh:mm)", labelpad=12)
        axis.set_ylabel("Indexed close (index points; start = 100)", labelpad=12)
        style_axis(axis)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.975, 0.925), ncol=3, frameon=False)
    figure.text(0.09, 0.095, "156 observations per ETF; 468 observed cells; 0 missing cells in the collected panel.", fontsize=10)
    figure.text(0.09, 0.048, "Separate sessions preserve the overnight gap. Lines use actual prices; no forward fill. Small collection pilot.", fontsize=9, color="#596C7D")
    save(figure, "G16.png")
    detail = stats(prices, manifest)
    detail["session_observations"] = {str(session): int((local.date == session).sum()) for session in sessions}
    return {
        "id": "G16", "title": "Equity ETF collection pilot: two separated intraday sessions",
        "group": "Additional market collection pilots", "figure": "G16.png",
        "equations": [r"I_{i,t}=100\,P_{i,t}/P_{i,t_0}"],
        "explanation": "Each line divides an ETF close by its first recorded October 1 close and starts at 100, making their relative price changes comparable. The separate session panels leave the overnight interval unconnected while preserving actual intraday clock times.",
        "method": "Read the frozen, adjusted Yahoo/yfinance SPY, QQQ and IWM 5-minute panel for October 1–2, 2026 and convert UTC labels to New York time. Plot actual observations without forward filling, with one common initial normalization across both sessions.",
        "finding": "The panel has 156 observations per ETF (78 in each session), 468 observed cells and zero missing cells. From the first recorded close to the last, SPY rose 0.64%, QQQ 0.90% and IWM 1.20%; these are descriptive changes within this small collection pilot.",
        "limitations": "Two sessions do not validate an intraday signal; provider bar-completion timing, adjustment history and broader coverage need further audit before forecasting.",
        "sources": sources(folder, provenance), "stats": detail,
    }


def fx_chart():
    prices, manifest, provenance, folder = panel("fx")
    levels = indexed(prices)
    figure, axis = plt.subplots(figsize=(12, 6))
    figure.subplots_adjust(left=0.09, right=0.83, bottom=0.25, top=0.78)
    figure.text(0.09, 0.935, "G17  |  ECB currency reference-rate pilot", fontsize=19, weight="bold")
    figure.text(0.09, 0.87, "2025 daily reference rates • foreign-currency units per EUR", fontsize=11)
    for color, symbol in zip(COLORS, prices.columns):
        axis.plot(prices.index, levels[symbol], color=color, linewidth=1.55, label=symbol)
    style_axis(axis)
    axis.set_ylabel("Indexed quote (index points; start = 100)", labelpad=12)
    axis.set_xlabel("Reference date (2025)", labelpad=12)
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.set_xlim(prices.index[0], prices.index[-1])
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False, title="Quote currency")
    figure.text(0.09, 0.095, "255 reference dates × 8 currencies; 2,040 observed cells; 0 missing cells in the collected panel.", fontsize=10)
    figure.text(0.09, 0.048, "Higher quotes mean more currency units per EUR. These are reference-rate movements, not investment returns.", fontsize=9, color="#596C7D")
    save(figure, "G17.png")
    return {
        "id": "G17", "title": "ECB reference-rate pilot: movements in eight currency quotes",
        "group": "Additional market collection pilots", "figure": "G17.png",
        "equations": [r"Q_{c,t}=\mathrm{units\ of\ currency}\ c\,/\,\mathrm{EUR}", r"I_{c,t}=100\,Q_{c,t}/Q_{c,t_0}", r"X_{c,t}=1/Q_{c,t}"],
        "explanation": "The ECB quotes the number of foreign-currency units exchanged for one euro; a rising line therefore means more units of that currency per euro. Reversing the quote requires its reciprocal, so this chart is not a direct currency investment-performance calculation.",
        "method": "Read the frozen official ECB panel for January 2–December 31, 2025 and index each quote to its first reference date. Use the 255 published reference dates as provided, without manufacturing weekend or holiday observations.",
        "finding": "All eight currencies have 255 observations (2,040 cells; zero missing). First-to-last quote changes range from SEK −5.26% to USD +13.85%, with JPY +13.61%; they measure movements against the common EUR numeraire.",
        "limitations": "Informational reference rates are not executable quotes or synchronized midnight prices; the common EUR numeraire also influences cross-currency dependence.",
        "sources": sources(folder, provenance), "stats": stats(prices, manifest),
    }


def crypto_chart():
    prices, manifest, provenance, folder = panel("crypto")
    levels = indexed(prices)
    figure, axis = plt.subplots(figsize=(12, 6))
    figure.subplots_adjust(left=0.09, right=0.975, bottom=0.25, top=0.78)
    figure.text(0.09, 0.935, "G18  |  Coinbase crypto collection pilot", fontsize=19, weight="bold")
    figure.text(0.09, 0.87, "BTC-USD, ETH-USD and SOL-USD • October 1, 2026 • 5-minute candles", fontsize=11)
    for color, symbol in zip(COLORS, prices.columns):
        axis.plot(prices.index, levels[symbol], color=color, linewidth=1.8, label=symbol)
    style_axis(axis)
    axis.set_ylabel("Indexed close (index points; start = 100)", labelpad=12)
    axis.set_xlabel("Candle start time (UTC); bar completes 5 minutes later", labelpad=12)
    axis.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 4, 8, 12, 16, 20], tz=prices.index.tz))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=prices.index.tz))
    axis.set_xlim(prices.index[0], prices.index[-1])
    axis.legend(loc="upper left", frameon=False, ncol=3)
    figure.text(0.09, 0.095, "288 candles per product; 864 observed cells; 0 missing cells on the declared 5-minute grid.", fontsize=10)
    figure.text(0.09, 0.048, "One UTC day on one venue. Missing candles would remain NaN; no forecast or trading claim is tested.", fontsize=9, color="#596C7D")
    save(figure, "G18.png")
    detail = stats(prices, manifest)
    detail["minimum_bar_close_availability_delay_seconds"] = 300
    detail["provider_publication_latency_measured"] = False
    return {
        "id": "G18", "title": "Coinbase collection pilot: one UTC day of three crypto products",
        "group": "Additional market collection pilots", "figure": "G18.png",
        "equations": [r"I_{i,t}=100\,P_{i,t}/P_{i,t_0}", r"t_{\mathrm{available}}\geq t_{\mathrm{start}}+5\,\mathrm{minutes}"],
        "explanation": "Each product's close is indexed to its first recorded candle close to compare relative movements. Labels denote candle starts; a close cannot be known before its five-minute bar ends, and provider publication latency was not measured.",
        "method": "Read the frozen Coinbase Exchange BTC-USD, ETH-USD and SOL-USD 5-minute closes for October 1, 2026 on the declared UTC grid. Preserve missing buckets as missing and compare first-to-last observed closes without fitting a model.",
        "finding": "All three products have 288 candles (864 cells; zero missing). From the first recorded close to the last, BTC-USD rose 1.65%, ETH-USD 0.86% and SOL-USD 0.39%; this confirms a usable bounded collection, not a forecasting edge.",
        "limitations": "One day, three products and one spot venue cannot establish market-wide structure or predictive value; publication latency is unmeasured, and the plotted changes exclude the return from the prior day's close to the first candle close.",
        "sources": sources(folder, provenance), "stats": detail,
    }


def workflow_tree():
    """Draw a compact hierarchy with numbered stages and an explicit future box."""
    figure, axis = plt.subplots(figsize=(13.5, 7.6))
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.985)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x, y, width, height, text, *, fill="white", edge=TEAL, textcolor=NAVY,
            fontsize=10.3, bold=False, dashed=False):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.009,rounding_size=0.012",
                              facecolor=fill, edgecolor=edge, linewidth=1.3,
                              linestyle="--" if dashed else "-")
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize,
                  color=textcolor, weight="bold" if bold else "normal", linespacing=1.55)

    def arrow(start, end, *, dashed=False, color=TEAL):
        axis.annotate("", xy=end, xytext=start,
                      arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.25,
                                  "linestyle": "--" if dashed else "-", "shrinkA": 3, "shrinkB": 3})

    box(0.32, 0.86, 0.36, 0.10, "Spectral market research\nReproduce → explain → validate",
        fill=NAVY, edge=NAVY, textcolor="white", fontsize=14, bold=True)
    centers = [0.125, 0.375, 0.625, 0.875]
    axis.plot([0.5, 0.5], [0.86, 0.815], color=TEAL, linewidth=1.3)
    axis.plot([centers[0], centers[-1]], [0.815, 0.815], color=TEAL, linewidth=1.3)
    for center in centers:
        arrow((center, 0.815), (center, 0.765))
    headers = ["DATA · stage 1", "MEASUREMENT · stages 2–4", "VALIDATION · stage 5", "OUTPUTS · stage 6"]
    for x, title in zip([0.015, 0.265, 0.515, 0.765], headers):
        box(x, 0.69, 0.22, 0.075, title, fill="#EAF2F5", fontsize=10.2, bold=True)

    arrow((0.125, 0.69), (0.125, 0.60))
    box(0.015, 0.465, 0.22, 0.13, "1. Freeze the main dataset\nS&P daily prices + provenance\nImmutable cache + local catalog")
    arrow((0.125, 0.465), (0.125, 0.41))
    box(0.015, 0.275, 0.22, 0.13, "Additional collection pilots\nETF intraday · ECB FX daily\nCoinbase crypto intraday")

    arrow((0.375, 0.69), (0.375, 0.605))
    box(0.265, 0.495, 0.22, 0.105, "2. Prepare observations\nPrices → arithmetic returns\n60 rows → common clean sample", fontsize=10)
    arrow((0.375, 0.495), (0.375, 0.45))
    box(0.265, 0.34, 0.22, 0.105, "3. Construct the graph\nPearson C → positive W\nDegrees D → L = D − W", fontsize=10)
    arrow((0.375, 0.34), (0.375, 0.295))
    box(0.265, 0.16, 0.22, 0.13, "4. Measure structure\nRaw λ₂ · q · normalized λ₂\nFiedler vector · gap · localization", fontsize=10)

    arrow((0.625, 0.69), (0.625, 0.60))
    box(0.515, 0.30, 0.22, 0.295, "5. Check the evidence\nOriginal session references\nSynthetic mathematical controls\nResampling stability\nData-quality flags", fontsize=10.5)

    arrow((0.875, 0.69), (0.875, 0.60))
    box(0.765, 0.465, 0.22, 0.13, "6. Report actual findings\nNumbered graphs + equations\nDescriptive results + limitations")
    arrow((0.875, 0.465), (0.875, 0.37), dashed=True, color=ORANGE)
    box(0.765, 0.20, 0.22, 0.165, "PLANNED NEXT STAGE\nNext-20-session variance\nChronological out-of-sample tests\nStrong risk-statistic baselines",
        fill="#FFF6EA", edge=ORANGE, fontsize=10, dashed=True)
    figure.text(0.035, 0.035, "Stages 1 to 6 show the generation sequence; dashed orange marks future work. Layout positions have no measurement axes.",
                 fontsize=10, color="#596C7D")
    figure.savefig(FIGURES / "process_tree.png", dpi=180, facecolor="white")
    figure.savefig(FIGURES / "process_tree.svg", facecolor="white")
    plt.close(figure)
    return {
        "title": "How the research generates and validates its findings",
        "description": "The grouped tree follows datasets, preparation, graph construction, measurements, validation and reporting. The orange next-20-session forecasting experiment is planned, rather than a completed predictive result.",
        "figure": "process_tree.png", "vector_figure": "process_tree.svg",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Refresh figures without rewriting saved numerical metadata")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    graphs = [etf_chart(), fx_chart(), crypto_chart()]
    tree = workflow_tree()
    if args.render_only:
        print("Refreshed G16-G18 and workflow; saved numerical metadata unchanged")
        return
    artifact = {"graphs": graphs, "tree": tree, "data_policy": "Existing registered frozen snapshots only; no downloads or data changes."}
    (OUTPUT / "pilots.json").write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved G16–G18, process_tree.png/.svg and pilots.json from existing frozen panels.")


if __name__ == "__main__":
    main()
