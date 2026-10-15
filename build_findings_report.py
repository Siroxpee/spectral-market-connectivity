"""Assemble the completed charts into an offline report and PDF input.

Run after report_charts_overview.py, report_charts_structure.py and
report_charts_pilots.py. This script reads frozen results; it never downloads.
"""

import base64
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/findings"

# These captions explain every displayed panel and distinguish measurement
# scales from ticker/date categories. They add no new numerical calculation.
AXIS_GUIDE = {
    "G01": "X is the final calendar date of each 60-return window. Y is raw lambda2, a unitless eigenvalue of that window's positive-correlation Laplacian; higher values mean stronger connectivity on its existing weight scale.",
    "G02": "Both x axes show window end dates. Top y is the unitless scale ratio q; bottom y is unitless normalized lambda2. The two panels use different definitions and vertical scales.",
    "G03": "X is window end date. Y is a unitless average across unordered stock pairs: the navy line includes negative correlations, while the teal line replaces negative values by zero. Shading shows the difference.",
    "G04": "Top and middle x axes show window end dates; y counts retained and removed stocks. Bottom x shows each source close date; y is the percentage of all 503 stocks with an observed close that day.",
    "G05": "Each dot is one window. Left x is mean raw correlation; right x is mean clipped graph weight. Both y axes show raw lambda2. All three measurements are unitless; dates are not the horizontal coordinate.",
    "G06": "Both x axes show window end dates. Top y compares minimum weighted degree and raw lambda2, both unitless sums on the graph's weight scale. Bottom y is their unitless Rayleigh-bound fraction; one denotes the upper bound.",
    "G07": "Both x axes show window end dates. Top y is unitless Fiedler localization IPR and the 1/n reference. Bottom y is the unitless gap between the third and second Laplacian eigenvalues.",
    "G08": "Left x is the percentage of source dates with an observed close; y lists ticker categories. Right x is arithmetic return in percent; y lists ticker-and-date categories. Bar order is a ranking, not elapsed time.",
    "G09": "Both x axes are calendar observation dates. Top y is FERG's close in index points, with the preceding close set to 100. Bottom y is each close-to-close arithmetic return in percent; orange marks exact zero returns.",
    "G10": "Both rows refer to selected window dates. Left x is unitless weighted degree; right x is unitless squared Fiedler-vector mass on a logarithmic scale. Every y axis lists ticker categories; each panel independently ranks its ten stocks.",
    "G11": "Each x axis categorizes the two window end dates. Y shows raw lambda2, q, or normalized lambda2, all unitless. Raw and q panels use logarithmic y scales. Bar color distinguishes the original graph from the separate node-removal sensitivity.",
    "G12": "X is column-stock position and y is row-stock position, both integer indices. Position zero is the weakest ticker; the others are alphabetical. The shared color scale is unitless Pearson correlation from minus one to one, not a sector classification.",
    "G13": "Left x is unitless common edge weight c; left y is raw lambda2 and right y is q. Right x identifies synthetic graphs A and B; y shows raw lambda2 or q. All displayed metric values are unitless.",
    "G14": "Each x axis categorizes the two reference market windows, not successive simulation times. Left y is unitless raw lambda2; right y is unitless q. Boxes summarize Gaussian replicates; orange stars locate the corresponding original observations on the same scales.",
    "G15": "Each x axis categorizes the two original window dates. Y panels show raw lambda2, q, normalized lambda2 and IPR, all unitless. Boxes summarize joint block resamples; orange stars show original values. Counts below each category identify available replicates.",
    "G16": "Both x axes show New York clock time in EDT (hours:minutes), one panel per session. Both y axes show index points relative to the first October 1 close, set to 100; 101 means a one-percent rise from that starting close.",
    "G17": "X is the ECB reference date during 2025. Y is index points for each foreign-currency-units-per-EUR quote, with its first reference quote set to 100. A higher line means more currency units per euro, not a currency investment return.",
    "G18": "X is candle-start clock time in UTC on October 1, 2026. Y is close-price index points, with the first recorded candle close set to 100. A candle's close is unavailable before its five-minute interval finishes.",
}


def _image_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _equation_image(equation, target):
    equation = equation.strip().strip("$")
    equation = re.sub(r"\\le\b", r"\\leq", equation)
    equation = re.sub(r"\\ge\b", r"\\geq", equation)
    equation = re.sub(r"\\mathcal\s+([A-Za-z])", r"\\mathcal{\1}", equation)
    equation = equation.replace(r"\bmod", r"\ \mathrm{mod}\ ")
    figure = plt.figure(figsize=(9, 0.6), dpi=180)
    figure.text(0.01, 0.5, "$" + equation + "$", fontsize=18,
                color="#17324D", va="center")
    figure.savefig(target, bbox_inches="tight", pad_inches=0.08, transparent=True)
    plt.close(figure)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def assemble_content():
    OUT.mkdir(parents=True, exist_ok=True)
    equations_dir = OUT / "equations"
    equations_dir.mkdir(exist_ok=True)
    graphs, tree = [], None
    for name in ("overview.json", "structure.json", "pilots.json"):
        data = json.loads((OUT / name).read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("graphs", [])
        graphs.extend(items)
        if isinstance(data, dict) and data.get("tree"):
            tree = data["tree"]
    graphs.sort(key=lambda item: item["id"])
    expected = [f"G{i:02d}" for i in range(1, 19)]
    if [graph["id"] for graph in graphs] != expected:
        raise ValueError("The report requires exactly G01 through G18 once each.")
    for graph in graphs:
        path = OUT / "figures" / Path(graph["figure"]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        graph["image_path"] = str(path)
        graph["figure_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        graph["axis_explanation"] = AXIS_GUIDE[graph["id"]]
        audit = path.with_suffix(".axes.json")
        if not audit.is_file():
            raise FileNotFoundError("Render updated labels before assembling: " + str(audit))
        graph["axis_labels"] = json.loads(audit.read_text(encoding="utf-8"))
        for field in ("title", "group", "explanation", "method", "finding", "limitations", "sources"):
            if field not in graph:
                raise ValueError(f"{graph['id']} is missing {field}.")
        rendered = []
        for number, equation in enumerate(graph.get("equations", []), 1):
            target = equations_dir / f"{graph['id']}-{number}.png"
            _equation_image(equation, target)
            rendered.append(str(target))
        graph["equation_images"] = rendered
    if tree is None:
        tree = {"title": "The research process", "figure": "process_tree.png",
                "description": "Datasets feed measurements, validation, and the findings report. The forecast branch is planned."}
    tree["image_path"] = str(OUT / "figures" / Path(tree["figure"]).name)
    tree["axis_explanation"] = "Axes do not apply: box positions organize workflow stages and branches, not measured x/y quantities or physical distances. Arrows show process relationships."
    if not Path(tree["image_path"]).is_file():
        raise FileNotFoundError(tree["image_path"])

    summary = json.loads((ROOT / "output/reproduction/summary.json").read_text())
    inventory = json.loads((ROOT / "output/local_validation/dataset_inventory.json").read_text())
    checks = json.loads((ROOT / "output/local_validation/checks_run.json").read_text())
    features = pd.read_csv(ROOT / "output/reproduction/features.csv")
    bound = features.num_stocks / (features.num_stocks - 1) * features.min_degree
    fraction = features.lambda2 / bound
    stats = {
        "windows": len(features),
        "correlation_raw": summary["lambda2_mean_raw_correlation"],
        "correlation_min_degree": float(features.lambda2.corr(features.min_degree)),
        "median_ipr": float(features.fiedler_ipr.median()),
        "ipr_over_090_count": int((features.fiedler_ipr > .9).sum()),
        "ipr_over_090_fraction": float((features.fiedler_ipr > .9).mean()),
        "median_bound_fraction": float(fraction.median()),
        "bound_fraction_over_098_count": int((fraction > .98).sum()),
    }
    if summary["reference_comparison"]["status"] != "matched" or checks["status"] != "passed":
        raise ValueError("Underlying reproduction or existing checks did not pass.")
    findings = [
        {"title": "The original daily result reproduces",
         "text": f"All {len(features):,} windows match the live notebook's dates and stock counts. The correlation of raw lambda2 with mean raw correlation is {stats['correlation_raw']:.15f}; maximum lambda2 disagreement is 1.99e-13.", "graphs": ["G01", "G05"]},
        {"title": "A weak individual stock often dominates raw lambda2",
         "text": f"Raw lambda2 and minimum degree have a descriptive correlation of {stats['correlation_min_degree']:.4f}. The median Fiedler concentration IPR is {stats['median_ipr']:.4f}, and {stats['ipr_over_090_count']:,} of {len(features):,} windows have IPR above 0.9. This points to concentrated weak-node modes rather than a broad partition in many windows.", "graphs": ["G06", "G07", "G10"]},
        {"title": "The March 2020 low needs a price-history check",
         "text": "On March 20, 2020, raw lambda2 is 1.299549 despite mean raw correlation of 0.641949. The weak-node mode centers on FERG, whose sixty-return window contains 34 exact zeros. Keeping these prices intact, we separate the graph mechanism from any unresolved ticker, venue, or adjustment explanation.", "graphs": ["G09", "G10", "G11"]},
        {"title": "Scale and structure answer different questions",
         "text": "On October 21, 2022, lambda2 is 110.654597 and q is 0.431172, versus 0.004179 in the March 2020 window. Mean correlation alone cannot explain this contrast. The synthetic examples show why scaling, bottlenecks and clipping must be distinguished.", "graphs": ["G02", "G03", "G12", "G13", "G14"]},
        {"title": "The additional markets remain short collection pilots",
         "text": "The archive holds one long stock panel and three prepared pilots: two equity sessions at five minutes, one year of daily FX reference rates, and one UTC crypto day at five minutes. Their coverage checks pass, but these spans do not support long-horizon conclusions.", "graphs": ["G16", "G17", "G18"]},
    ]
    glossary = [
        (r"r_{i,t}=P_{i,t}/P_{i,t-1}-1", "Arithmetic return: today's close divided by the previous close, minus one. Missing closes do not become zero returns."),
        (r"C_{ij}=\operatorname{corr}(r_i,r_j)", "Pearson correlation is calculated from the same sixty return observations for every retained stock pair."),
        (r"W_{ij}=\max(C_{ij},0),\quad W_{ii}=0", "Negative correlations are removed from this particular positive-coupling graph; self-edges are zero."),
        (r"d_i=\sum_j W_{ij},\quad D=\operatorname{diag}(d_i),\quad L=D-W", "Weighted degree sums a stock's positive links. The unnormalized Laplacian combines those degrees with the edge weights."),
        (r"\lambda_2=\min_{v^T\mathbf{1}=0,\ v^Tv=1}v^TLv", "Lambda2 is the lowest graph disagreement energy among unit vectors orthogonal to the constant vector."),
        (r"\mu_C=\frac{2}{n(n-1)}\sum_{i<j}C_{ij},\quad \mu_W=\frac{2}{n(n-1)}\sum_{i<j}W_{ij}", "Mean raw correlation keeps negative pairs. Mean clipped weight includes every pair, including zero-weight pairs."),
        (r"q=\frac{\lambda_2}{n\mu_W}", "For positive mean edge weight, the scale ratio removes a uniform multiplication of all edges. It does not remove weak-node effects, composition changes or estimation noise."),
        (r"\mathcal{L}=I-D^{-1/2}WD^{-1/2}", "For positive degrees, the normalized Laplacian adjusts for weighted degree; its second eigenvalue differs from raw lambda2. The starter reports it for connected graphs, as all baseline graphs are here."),
        (r"\mathrm{IPR}=\sum_i v_i^4,\quad \Delta=\lambda_3-\lambda_2", "For a unit Fiedler vector, IPR near one means concentration on a few stocks; a uniform-magnitude vector has IPR 1/n. The eigengap checks numerical separation."),
        (r"\lambda_2\leq\frac{n}{n-1}\min_i d_i", "A single-node contrast gives an upper bound on lambda2, explaining why one weak stock can constrain the graph's result."),
        (r"B_{i,t}=100\frac{P_{i,t}}{P_{i,0}}", "Rebasing makes price or quote changes comparable visually. It does not convert FX reference quotes into investment returns."),
    ]
    glossary_items = []
    for i, (equation, explanation) in enumerate(glossary, 1):
        image = equations_dir / f"glossary-{i:02d}.png"
        _equation_image(equation, image)
        glossary_items.append({"equation": equation, "explanation": explanation, "image_path": str(image)})
    content = _plain({
        "title": "Spectral market findings",
        "subtitle": "What the frozen data show, how each graph is calculated, and what remains to test",
        "date_label": "8 October 2026 Pacific",
        "claim_scope": "Descriptive market measurements with synthetic and resampling diagnostics. The forecasting branch is planned.",
        "graphs": graphs, "tree": tree, "findings": findings,
        "glossary": glossary_items, "statistics": stats,
        "inventory": inventory, "reproduction": summary, "checks": checks,
        "limitations": [
            "The stock panel uses the submitted contemporary membership list looking backward; some assets have shorter histories. It is not a point-in-time historical S&P membership panel.",
            "Original Yahoo adjustment settings, download time, historical ticker mappings and historical download package versions remain unverified. The report preserves the supplied prices and flags questions rather than repairing them without evidence.",
            "Adjacent sixty-observation windows overlap by fifty-nine observations. The plotted associations are descriptive and have no independent-sample significance claim.",
            "A separated Fiedler eigenvector is numerically identifiable, not automatically economically stable. The block-resampling and Gaussian controls depend on their stated assumptions and are exploratory.",
            "The small ETF, FX and crypto pilots are separate panels with different clocks and units. They have not been merged into a common market network.",
        ],
        "next_steps": [
            "Reconcile FERG's flat saved observations and the flagged corporate actions using documented price/adjustment histories.",
            "Evaluate whether localization survives longer windows, a smaller declared universe, factor controls and resampling choices.",
            "Then freeze a next-twenty-session variance target, strong baselines, mature-label chronological training and an honest holdout before any predictive comparison.",
        ],
    })
    (OUT / "report_content.json").write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    return content


def write_html(content):
    esc = html.escape
    groups = list(dict.fromkeys(graph["group"] for graph in content["graphs"]))
    findings = ''.join(f'<article class="finding"><h3>{esc(f["title"])}</h3><p>{esc(f["text"])}</p><p class="refs">'+', '.join(f'<a href="#{g}">{g}</a>' for g in f['graphs'])+'</p></article>' for f in content['findings'])
    rows = ''.join(f'<tr><td><a href="#{g["id"]}">{g["id"]}</a></td><td>{esc(g["title"])}</td><td>{esc(g["explanation"])}</td><td>{esc(g["axis_explanation"])}</td></tr>' for g in content['graphs'])
    cards = []
    for graph in content['graphs']:
        equations = ''.join(f'<img class="equation" src="{_image_url(path)}" alt="{esc(eq)}">' for path, eq in zip(graph['equation_images'], graph['equations']))
        sources = '<br>'.join(esc(str(source)) for source in graph['sources'])
        cards.append(f'''<section class="graph" id="{graph['id']}" data-group="{esc(graph['group'])}">
        <div class="eyebrow">{graph['id']} / {esc(graph['group'])}</div><h2>{esc(graph['title'])}</h2>
        <p class="lead">{esc(graph['explanation'])}</p>
        <img class="chart" src="{_image_url(graph['image_path'])}" alt="{esc(graph['title'])}" tabindex="0">
        <p class="axes"><strong>Reading the axes:</strong> {esc(graph['axis_explanation'])}</p>
        <div class="detail-grid"><div><h3>Equation</h3>{equations}<h3>How we get the graph</h3><p>{esc(graph['method'])}</p></div>
        <div><h3>Finding</h3><p>{esc(graph['finding'])}</p><h3>Interpretation limits</h3><p>{esc(graph['limitations'])}</p></div></div>
        <details><summary>Source and calculation record</summary><p class="source">{sources}</p><pre>{esc(json.dumps(graph.get('stats',{}),indent=2))}</pre></details></section>''')
    glossary = ''.join(f'<div class="glossary-item"><img class="equation" src="{_image_url(item["image_path"])}" alt="{esc(item["equation"])}"><p>{esc(item["explanation"])}</p></div>' for item in content['glossary'])
    options = '<option value="all">All 18 graphs</option>'+''.join(f'<option value="{esc(group)}">{esc(group)}</option>' for group in groups)
    limitations = ''.join(f'<li>{esc(item)}</li>' for item in content['limitations'])
    next_steps = ''.join(f'<li>{esc(item)}</li>' for item in content['next_steps'])
    stock = content['reproduction']
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(content['title'])}</title>
    <style>
    :root{{--navy:#17324d;--teal:#007f86;--orange:#d57b27;--muted:#5e6c79;--line:#dce4e9;--paper:#fff;--bg:#f3f6f8}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--navy);font:16px/1.6 Arial,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 24px 80px}}h1,h2{{font-family:Georgia,serif;line-height:1.15}}h1{{font-size:48px;margin:10px 0 16px}}h2{{font-size:29px;margin:10px 0 18px}}h3{{font-size:17px;margin:18px 0 8px}}p{{margin:0 0 14px}}a{{color:var(--teal)}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:1.6px;font-weight:bold;color:var(--teal)}}.lead{{font-size:18px;max-width:920px}}.toolbar{{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;padding:12px 24px;background:#17324df2;color:white;backdrop-filter:blur(8px)}}.toolbar a{{color:white;text-decoration:none;font-size:14px}}.toolbar select,.toolbar button{{padding:7px 10px;border:1px solid #647c8d;border-radius:5px;background:white;color:var(--navy);font:inherit;font-size:14px}}.toolbar .spacer{{flex:1}}.section,.graph{{background:white;border:1px solid var(--line);border-radius:12px;padding:28px;margin:26px 0;scroll-margin-top:90px}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}}.metric{{background:white;border:1px solid var(--line);border-top:4px solid var(--teal);border-radius:8px;padding:20px}}.metric strong{{display:block;font-size:28px;line-height:1.2}}.metric span{{font-size:13px;color:var(--muted)}}.finding-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.finding{{background:#f4f8fa;border-left:3px solid var(--teal);padding:16px 20px}}.finding h3{{margin-top:0}}.finding:last-child{{grid-column:1/-1}}.refs{{font-size:13px;margin:0}}.chart{{width:100%;display:block;border-radius:6px;cursor:zoom-in}}.tree{{max-height:740px;width:100%;object-fit:contain}}.detail-grid{{display:grid;grid-template-columns:1fr 1fr;gap:32px;border-top:1px solid var(--line);margin-top:16px}}.equation{{max-width:100%;max-height:68px;object-fit:contain;object-position:left;display:block;margin:8px 0 12px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid var(--line)}}th{{color:var(--muted)}}td:first-child{{white-space:nowrap}}.table-wrap{{overflow:auto}}.source,pre{{font-size:12px;overflow-wrap:anywhere;color:var(--muted)}}pre{{white-space:pre-wrap}}details{{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}}summary{{cursor:pointer;font-size:13px;color:var(--muted)}}.glossary-item{{border-bottom:1px solid var(--line);padding:12px 0}}.notice{{border-left:3px solid var(--orange);padding:12px 16px;background:#fff8f0}}dialog{{border:0;width:95vw;max-width:none;background:#fff;border-radius:8px}}dialog::backdrop{{background:#000b}}dialog img{{width:100%}}dialog button{{position:sticky;top:0;float:right;padding:8px 16px}}.small{{font-size:13px;color:var(--muted)}}.graph.hidden{{display:none}}
    @media(max-width:700px){{main{{padding:25px 12px}}h1{{font-size:36px}}.section,.graph{{padding:18px}}.finding-grid,.detail-grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr}}.toolbar{{padding:10px 12px;flex-wrap:wrap}}.toolbar .spacer{{display:none}}}}
    @media print{{body{{background:white;font-size:10pt}}.toolbar,details,dialog{{display:none}}main{{padding:0;max-width:none}}.section,.graph{{border:0;border-radius:0;break-before:page;padding:0;margin:0}}.graph.hidden{{display:block}}.chart{{max-height:100mm;object-fit:contain}}.detail-grid{{gap:15px}}h1{{font-size:28pt}}h2{{font-size:20pt}}.finding-grid{{display:block}}.finding{{margin:10px 0}}}}
    </style></head><body><div class="toolbar"><a href="#top">Findings</a><a href="#process">Process tree</a><a href="#graph-list">Graph list</a><a href="#equations">Equations</a><span class="spacer"></span><select id="filter">{options}</select><a href="../pdf/Spectral_Market_Findings.pdf">PDF</a><button onclick="window.print()">Print</button></div>
    <main id="top"><div class="eyebrow">Research report / {esc(content['date_label'])}</div><h1>{esc(content['title'])}</h1><p class="lead">{esc(content['subtitle'])}</p><p>{esc(content['claim_scope'])}</p>
    <div class="metrics"><div class="metric"><strong>2,643</strong><span>Daily windows reproduced</span></div><div class="metric"><strong>18 graphs</strong><span>Each with equations, data method and finding</span></div><div class="metric"><strong>$0</strong><span>Additional project spend</span></div></div>
    <section class="section"><h2>The findings</h2><div class="finding-grid">{findings}</div></section>
    <section class="section" id="process"><div class="eyebrow">The whole process</div><h2>{esc(content['tree']['title'])}</h2><p>{esc(content['tree']['description'])}</p><p><strong>Axes:</strong> {esc(content['tree']['axis_explanation'])}</p><img class="tree chart" src="{_image_url(content['tree']['image_path'])}" alt="Grouped workflow from datasets through validation and findings, with the forecasting branch planned"></section>
    <section class="section"><h2>The datasets behind the graphs</h2><p>One dataset means a declared asset universe, interval and historical span. The archive contains one long stock panel and three prepared collection pilots.</p><div class="table-wrap"><table><thead><tr><th>Panel</th><th>Assets</th><th>Coverage</th><th>Observed values</th></tr></thead><tbody><tr><td>Original stock closes, daily</td><td>503 symbols</td><td>4 Jan 2016 - 2 Oct 2026; 2,703 dates</td><td>1,315,118</td></tr><tr><td>ETF closes, five minutes</td><td>SPY / QQQ / IWM</td><td>1-2 Oct 2026; 156 timestamps</td><td>468</td></tr><tr><td>FX reference rates, daily</td><td>Eight currencies versus EUR</td><td>2025; 255 dates</td><td>2,040</td></tr><tr><td>Crypto closes, five minutes</td><td>BTC / ETH / SOL versus USD</td><td>1 Oct 2026 UTC; 288 timestamps</td><td>864</td></tr></tbody></table></div><p class="small">Prepared total: 1,318,490 observations, plus 44,491 missing original-stock cells. The preserved broader ECB archive contains older history that is not plotted as an additional prepared panel.</p></section>
    <section class="section" id="graph-list"><div class="eyebrow">Numbered index</div><h2>All 18 graphs</h2><p>Click a graph number to see its equation, data calculation, axes and units, finding and interpretation limits. Click the image to enlarge it. Unitless means a number has no physical measurement unit; index points instead set a declared starting level to 100.</p><div class="table-wrap"><table><thead><tr><th>Graph</th><th>Title</th><th>What it shows</th><th>Axes and units</th></tr></thead><tbody>{rows}</tbody></table></div></section>
    {''.join(cards)}
    <section class="section" id="equations"><h2>The equations in plain language</h2>{glossary}</section>
    <section class="section"><h2>What remains to test</h2><ul>{limitations}</ul><h3>The next research steps</h3><ol>{next_steps}</ol><p class="notice">The process tree marks forecasting as planned. This report contains no scored future-variance model or trading experiment.</p></section>
    <section class="section"><h2>Evidence and reproducibility</h2><p>The original frozen snapshot was reloaded in the live kernel with zero differing numeric values. The daily references match exactly in dates and counts, and all twenty existing check groups pass.</p><p class="source">Data: original frozen notebook matrix; reproduced feature table; selected-window diagnostic tables; and three frozen provider pilots. Sources and calculation records appear below each graph.</p><details><summary>Original snapshot and execution versions</summary><pre>{esc(json.dumps({'snapshot_key':stock['snapshot_key'],'price_sha256':stock['snapshot_price_sha256'],'window_size':60,'step_size':1,'execution_versions':stock['package_versions']},indent=2))}</pre></details></section>
    <p class="small">All figures are local, reproducible artifacts. No network connection is required to read this report.</p></main><dialog id="zoom"><button id="close">Close</button><img id="zoom-img" alt="Enlarged research graph"></dialog>
    <script>const filter=document.getElementById('filter');filter.addEventListener('change',()=>document.querySelectorAll('.graph').forEach(g=>g.classList.toggle('hidden',filter.value!=='all'&&g.dataset.group!==filter.value)));document.querySelectorAll('a[href^="#G"]').forEach(a=>a.addEventListener('click',()=>{{filter.value='all';document.querySelectorAll('.graph').forEach(g=>g.classList.remove('hidden'))}}));const dlg=document.getElementById('zoom');document.querySelectorAll('.chart').forEach(img=>{{const open=()=>{{document.getElementById('zoom-img').src=img.src;dlg.showModal()}};img.addEventListener('click',open);img.addEventListener('keydown',e=>{{if(e.key==='Enter')open()}})}});document.getElementById('close').onclick=()=>dlg.close();dlg.addEventListener('click',e=>{{if(e.target===dlg)dlg.close()}});</script></body></html>'''
    target = OUT / "Findings_Report.html"
    target.write_text(page, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    write_html(assemble_content())
