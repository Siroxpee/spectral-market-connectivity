"""Draw the expanded research workflow offline; no data acquisition or analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY, TEAL, ORANGE = "#17324D", "#007F86", "#D57B27"
ROOT = Path(__file__).resolve().parent


def box(ax, x, y, width, height, title, body, *, color=TEAL, fill="#F3F8F9", size=10):
    """Coordinates describe the center of a rounded, two-part text card."""
    ax.add_patch(FancyBboxPatch(
        (x-width/2, y-height/2), width, height,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        facecolor=fill, edgecolor=color, linewidth=1.3, zorder=3))
    ax.text(x, y+height/2-0.16, title, ha="center", va="center",
            color=color, fontsize=size+1, fontweight="bold", zorder=4)
    ax.text(x, y-0.11, body, ha="center", va="center",
            color=NAVY, fontsize=size, linespacing=1.3, zorder=4)


def arrow(ax, start, end, *, color=TEAL, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                 mutation_scale=10, linewidth=1.2, color=color,
                 linestyle="--" if dashed else "-", zorder=2))


def fan(ax, sources, destinations, rail_y, *, color=TEAL):
    """Join branches on an explicit horizontal rail, then point into children."""
    points = [x for x, _ in sources+destinations]
    ax.plot([min(points), max(points)], [rail_y, rail_y], color=color, lw=1.2, zorder=2)
    for x, y in sources:
        ax.plot([x, x], [y, rail_y], color=color, lw=1.2, zorder=2)
    for x, y in destinations:
        arrow(ax, (x, rail_y), (x, y), color=color)


def draw_tree():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(11, 9), facecolor="white")
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 11); ax.set_ylim(0, 9); ax.axis("off")
    ax.text(5.5, 8.82, "Expanded acquisition and spectral research workflow",
            ha="center", va="center", fontsize=17, fontweight="bold")

    source_x = [1.45, 4.15, 6.85, 9.55]
    for x, title, body in zip(source_x,
        ["Financial prices", "Crypto candles", "Physical prices", "Grid operations"],
        ["Stocks, ETFs, FX, futures\nDeclared market and universe",
         "Venue and listed products\nActual available history",
         "Electricity and gas contracts\nSigned and zero prices retained",
         "Demand, generation, flows\nOperating context"]):
        box(ax, x, 8.21, 2.3, 0.68, title, body, size=9.4)
    box(ax, 5.5, 7.20, 9.0, 0.65, "Freeze observations and audit coverage",
        "Raw hashes + source identity, units, market, interval and time labels\nRequested vs. actual coverage; gaps and failures preserved; no silent source changes", size=10)
    fan(ax, [(x, 7.835) for x in source_x], [(5.5, 7.56)], 7.69)

    box(ax, 2.15, 6.22, 3.15, 0.73, "Financial / crypto return panels",
        "Arithmetic returns: $r_t=P_t/P_{t-1}-1$\nNo fill; declared interval gaps masked", size=10)
    box(ax, 5.70, 6.22, 3.15, 0.73, "Price / energy difference panels",
        "First differences: $\\Delta X_t=X_t-X_{t-1}$\nLevels kept separately; no price ratios", size=10)
    box(ax, 9.10, 6.22, 2.45, 0.73, "Operating context",
        "Keep units and time alignment\nNot an input to the price graph", color=ORANGE, fill="#FFF7EE", size=9.5)
    fan(ax, [(5.5, 6.84)], [(2.15, 6.62), (5.70, 6.62), (9.10, 6.62)], 6.74)

    box(ax, 4.15, 5.24, 7.55, 0.65, "One common cleaned 60-observation window",
        "Retain assets with complete finite data and sample standard deviation $s>10^{-12}$\nNo filling. Sixty observations have different durations at different intervals.", size=10)
    fan(ax, [(2.15, 5.815), (5.70, 5.815)], [(4.15, 5.60)], 5.70)
    box(ax, 4.15, 4.27, 7.55, 0.76, "Preserved positive-correlation baseline",
        "$C=\\mathrm{Pearson},\\quad W_{ij}=\\max(C_{ij},0),\\ W_{ii}=0,\\quad D=\\mathrm{diag}(W\\mathbf{1}),\\quad L=D-W$\nFull spectra of $C$ and $L$ at scheduled windows; selected alternative operators", color=NAVY, fill="#F1F5F9", size=10.3)
    arrow(ax, (4.15, 4.88), (4.15, 4.685))

    math_x = [1.45, 4.15, 6.85, 9.55]
    titles = ["Linear algebra", "Graphs / combinatorics", "Permutation group\nand matrix algebra", "GF(2) topology\nand geometry"]
    bodies = ["Eigenvalue scale, rank\nNormalized connectivity\nFiedler localization",
              "Degree and components\nThreshold graphs\nTriangles and graph cycles",
              "Relabeling invariance\nMatrix powers / closed walks\nCommutator diagnostic",
              "Clique $H_0,H_1$: up to 30 nodes\nCorrelation-distance MST\n$H_0$ merge distances"]
    for x, title, body in zip(math_x, titles, bodies):
        box(ax, x, 3.02, 2.3, 0.97, title, body, size=9.2)
    fan(ax, [(4.15, 3.855)], [(x, 3.54) for x in math_x], 3.70)

    box(ax, 5.5, 1.94, 9.65, 0.61, "Descriptive candidates for further investigation",
        "Dependence, scale, weak-node localization, changing composition and bottlenecks\nSelected complementary methods; these are not independent proofs of a financial edge", size=10)
    fan(ax, [(x, 2.50) for x in math_x], [(5.5, 2.28)], 2.39)

    future_x = [1.92, 5.50, 9.08]
    future_titles = ["Reconcile source and timing", "Test a chronological holdout", "Establish a physical route"]
    future_bodies = ["Price / contract identity and revisions\nAvailable-at-time information\nListing and corporate-action checks",
                     "Strong baselines; overlapping labels\nRobustness and net-cost evaluation\nPrediction remains unvalidated",
                     "Measured capacity and constraints\nLosses, settlement rules and fees\nFeasible transfer remains unvalidated"]
    for x, title, body in zip(future_x, future_titles, future_bodies):
        box(ax, x, 0.80, 3.12, 0.91, title, body, color=ORANGE, fill="#FFF7EE", size=9.5)
    fan(ax, [(5.5, 1.60)], [(x, 1.29) for x in future_x], 1.45, color=ORANGE)

    # Operating context bypasses the price-correlation graph. This dashed path
    # supplies future physical checks, not invented transmission-edge weights.
    ax.plot([10.36, 10.95, 10.95], [6.22, 6.22, 0.80],
            color=ORANGE, lw=1.2, linestyle="--", zorder=1)
    arrow(ax, (10.95, 0.80), (10.68, 0.80), color=ORANGE, dashed=True)
    ax.text(5.5, 0.15, "Current scope: exploratory description. Orange branches are required future validation; dashed path supplies operating context.",
            ha="center", va="center", fontsize=9, color=NAVY)
    output = ROOT / "output/expansion/figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output / ("expanded_process_tree." + extension)
        fig.savefig(path, dpi=160, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    draw_tree()
