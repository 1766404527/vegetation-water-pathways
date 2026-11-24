"""Draw submission Figures 1 and 2 from the study's documented workflow.

PNG files are rendered at 1000 dpi; PDF embeds TrueType fonts and SVG keeps
editable text. Effect decomposition follows the equations in sem.py.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as PlotPath


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK = "#253744"
LINE = "#172C51"
# Reuse the original manuscript's semantic palette: input, process, results,
# and integrated outcome. The three analysis branches remain parallel.
INPUT = "#C6ECAE"
PROCESS = "#F2E3BC"
RESULT = "#777DA7"
OUTCOME = "#D9D9D9"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "afm-framework-20261103",
    "savefig.facecolor": "white",
})


def canvas(width, height):
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, width, height, face="white", edge=LINE, lw=0.8):
    patch = Rectangle((x, y), width, height, facecolor=face,
                      edgecolor=edge, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=10, bold=False, ha="center", **kwargs):
    return ax.text(x, y, text, ha=ha, va="center", fontsize=size,
                   fontweight="bold" if bold else "normal", zorder=5,
                   linespacing=1.35, **kwargs)


def line(ax, points, lw=0.85, color=LINE):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=4)


def arrow(ax, start, end, lw=0.9, color=LINE, size=9):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>",
                            mutation_scale=size, linewidth=lw,
                            edgecolor=color, facecolor=color,
                            shrinkA=0, shrinkB=0, zorder=2)
    ax.add_patch(patch)
    return patch


def indirect_arrow(ax, curve, *, color, size=11, name=None):
    """Use the direct arrow's complete head, with dashes on the shaft only."""
    reference = FancyArrowPatch(path=curve, arrowstyle="-|>",
                                mutation_scale=size, linewidth=1.35,
                                edgecolor=color, facecolor=color, zorder=2)
    ax.add_patch(reference)
    # Resolve the arrow in the actual display transform before splitting it.
    # Both pieces then retain the same physical geometry in PNG/PDF/SVG.
    ax.figure.canvas.draw()
    outline = reference.get_path()
    starts = [i for i, code in enumerate(outline.codes)
              if code == PlotPath.MOVETO]
    assert len(starts) == 2, "Expected a shaft and one filled arrowhead"
    split = starts[1]
    reference.remove()
    shaft = PathPatch(PlotPath(outline.vertices[:split], outline.codes[:split]),
                      fill=False, edgecolor=color, linewidth=1.35,
                      linestyle=(0, (4, 2.8)), capstyle="round", zorder=2)
    head = PathPatch(PlotPath(outline.vertices[split:], outline.codes[split:]),
                     facecolor=color, edgecolor=color, linewidth=1.35,
                     linestyle="solid", joinstyle="round", zorder=2)
    if name:
        shaft.set_gid(name)
        head.set_gid(name + "-arrowhead")
    ax.add_patch(shaft)
    ax.add_patch(head)


def save(fig, number):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"Figure_{number}"
    fig.canvas.draw()
    # Catch accidental text overflow outside the full figure at authoring time.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for item in fig.axes[0].texts:
        extent = item.get_window_extent(renderer)
        assert bounds.contains(extent.x0, extent.y0), item.get_text()
        assert bounds.contains(extent.x1, extent.y1), item.get_text()
    fig.savefig(stem.with_suffix(".png"), dpi=1000)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Title": f"Figure {number}"})
    fig.savefig(stem.with_suffix(".svg"), metadata={"Title": f"Figure {number}"})
    plt.close(fig)


def figure_1():
    fig, ax = canvas(190, 170)
    label(ax, 7, 163, "Data sources", size=11, bold=True, ha="left")
    sources = [
        (7, "VOD and SM", "SMOS-IC V2"),
        (52, "PRE", "PERSIANN-CDR"),
        (97, "VPD", "ERA5-Land"),
        (142, "ET", "GLEAM v4.1a"),
    ]
    for x, variable, product in sources:
        box(ax, x, 137, 41, 19, face=INPUT)
        label(ax, x + 20.5, 150, variable, size=11, bold=True)
        label(ax, x + 20.5, 142.5, product, size=10)
        line(ax, [(x + 20.5, 137), (x + 20.5, 130)])
    line(ax, [(27.5, 130), (162.5, 130)])
    arrow(ax, (95, 130), (95, 125))

    box(ax, 7, 98, 176, 27, face=PROCESS)
    label(ax, 95, 119, "Shared preprocessing", size=11, bold=True)
    label(ax, 95, 110.8,
          "Quality control / Monthly aggregation / Alignment to the 25-km SMOS-IC grid", size=10)
    label(ax, 95, 103.4,
          "MODIS vegetation mask / Monthly standardized anomalies (June 2010-June 2023)", size=10)
    line(ax, [(95, 98), (95, 93)])
    line(ax, [(35, 93), (155, 93)])

    branches = [
        (7, "Statistical association",
         ["Pearson correlation", "Modified Mann-Kendall test", "Theil-Sen slopes"],
         ["Correlation and trend maps", "SMAP soil-moisture comparison"]),
        (67, "Dominant-factor attribution",
         ["Regularized random forests", "OOB skill screening", "Regional drop-column analysis"],
         ["RF importance and dominance", "Temporal-stability diagnostics"]),
        (127, "Pathway decomposition",
         ["Standardized OLS path model", "Direct, indirect and total effects", "Bootstrap uncertainty"],
         ["Path coefficients and dominance", "Forest-non-forest comparison"]),
    ]
    for x, heading, methods, outputs in branches:
        cx = x + 28
        arrow(ax, (cx, 93), (cx, 88))
        box(ax, x, 32, 56, 56, face=PROCESS)
        box(ax, x, 76, 56, 12, face=PROCESS)
        box(ax, x, 32, 56, 18.5, face=RESULT)
        label(ax, cx, 82, heading, size=10.1, bold=True)
        for y, text in zip((69, 62.5, 56), methods):
            label(ax, cx, y, text, size=9.6)
        for y, text in zip((44, 37.5), outputs):
            label(ax, cx, y, text, size=9.4, color="#000000")
        line(ax, [(cx, 32), (cx, 27)])
    line(ax, [(35, 27), (155, 27)])
    arrow(ax, (95, 27), (95, 21))
    box(ax, 7, 2, 176, 19, face=OUTCOME)
    label(ax, 95, 15.5, "Integrated interpretation", size=11, bold=True)
    label(ax, 95, 7.8,
          "Spatial heterogeneity / RF-path concordance / Vegetation water-stress responses", size=10)
    save(fig, 1)


def figure_2():
    fig, ax = canvas(180, 118)
    # This is an effect-decomposition schematic, matching Figure 11:
    # four direct effects and five complete mediated paths, not nine
    # individual regression terms. Neutral arrows do not assume signs.
    nodes = {"VPD": (19, 83), "SM": (64, 106), "PRE": (117, 106),
             "ET": (162, 83), "VOD": (90, 22)}
    node_width, node_height = 22, 11
    dark = "#202020"
    direct = [((19, 77.5), (82, 27.5)), ((64, 100.5), (86, 27.5)),
              ((117, 100.5), (94, 27.5)), ((162, 77.5), (98, 27.5))]
    for start, end in direct:
        arrow(ax, start, end, lw=1.35, color=dark, size=11)

    paths = [
        ("PRE-SM-VOD", [(106, 106), (97, 106), (89, 66), (90, 27.5)], "SM", 0.48, 16),
        ("ET-SM-VOD", [(162, 77.5), (161, 47), (126, 22), (101, 22)], "SM", 0.45, 16),
        ("VPD-SM-VOD", [(30, 83), (58, 83), (40, 35), (79, 25)], "SM", 0.34, 16),
        ("VPD-ET-VOD", [(19, 77.5), (20, 48), (43, 25), (79, 22)], "ET", 0.30, 16),
        ("VPD-ET-SM-VOD", [(12.5, 77.5), (9, 46), (36, 14), (79, 19)], "ET \u2192 SM", 0.56, 25),
    ]
    assert len(direct) == 4 and len(paths) == 5
    assert paths[-1][0] == "VPD-ET-SM-VOD"
    for name, points, mediator, t, capsule_width in paths:
        curve = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.CURVE4] * 3)
        indirect_arrow(ax, curve, color=dark, name=name)
        weights = ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)
        cx, cy = [sum(w*p[i] for w,p in zip(weights, points)) for i in (0, 1)]
        tangent = [3*(1-t)**2*(points[1][i]-points[0][i]) +
                   6*(1-t)*t*(points[2][i]-points[1][i]) +
                   3*t*t*(points[3][i]-points[2][i]) for i in (0, 1)]
        angle = math.degrees(math.atan2(tangent[1], tangent[0]))
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180
        ax.add_patch(Ellipse((cx, cy), capsule_width, 8.5, angle=angle,
                            facecolor="white", edgecolor=dark, linewidth=1, zorder=4))
        label(ax, cx, cy, mediator, size=10.5, bold=True, color=dark,
              fontstyle="italic", rotation=angle, rotation_mode="anchor")
    for text, (x, y) in nodes.items():
        box(ax, x - node_width / 2, y - node_height / 2,
            node_width, node_height, edge=LINE, lw=1)
        label(ax, x, y, text, size=13, bold=True, color="#111111")
    arrow(ax, (43, 6.5), (56, 6.5), lw=1.35, color=dark, size=10)
    label(ax, 59, 6.5, "Direct effect", size=10, ha="left", color=dark)
    legend_curve = PlotPath([(106, 6.5), (119, 6.5)],
                           [PlotPath.MOVETO, PlotPath.LINETO])
    indirect_arrow(ax, legend_curve, color=dark, size=10, name="indirect-legend")
    label(ax, 122, 6.5, "Indirect effect", size=10, ha="left", color=dark)
    save(fig, 2)


if __name__ == "__main__":
    figure_1()
    figure_2()
    print("Updated Figures 1 and 2: PNG (1000 dpi), PDF, SVG.")
