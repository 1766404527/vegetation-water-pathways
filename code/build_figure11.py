"""Rebuild Figure 11 from its editable SVG template and path-result tables.

The map and path-diagram layout is read from the editable SVG template.
Panel-b and panel-c labels are populated from the current forest
and non-forest CSV tables, and arrow colours are derived from coefficient
signs (blue positive, red negative).  This prevents manual transcription of
the displayed mean coefficients. The serial VPD-ET-SM-VOD effect is drawn as
a separate outer dashed curve in both panels, with a shared bottom legend.

The Python dependencies used to update the SVG (pandas and the standard XML
library) are already listed in ``requirements.txt``.  PNG export uses Microsoft
Edge in headless mode.  Use ``--skip-png`` when only the editable SVG is
required.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE / "result1" / "f_sem"

DEFAULT_TEMPLATE = RESULT_DIR / "Figure_11_template.svg"
DEFAULT_FOREST = RESULT_DIR / "f_SEM_ETnolag.csv"
DEFAULT_NONFOREST = RESULT_DIR / "nf_SEM_ETnolag.csv"
DEFAULT_SVG = RESULT_DIR / "Figure_11.svg"
DEFAULT_PNG = BASE / "figures" / "Figure_11.png"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

POSITIVE = "#83B1D3"
NEGATIVE = "#FA8070"
SERIAL_EFFECT = "VPD-ET-SM-VOD Indirect"

PANEL_C_FONT_SIZES = {"55": "72", "73": "86", "83": "92"}


# Each value is anchored by the unique transform of its text object in the SVG.
# Separate sign objects are retained where the design provides them; slots
# without a separate sign object use a signed string.
TEXT_SLOTS = {
    "forest": {
        "SM Direct": ("matrix(0.296933 0.954898 -0.954898 0.296933 891.28 1634)", None),
        "VPD Direct": ("matrix(0.79894 0.60141 -0.60141 0.79894 702.36 2137)", None),
        "PRE Direct": (
            "matrix(0.399908 -0.916555 0.916555 0.399908 1464.29 1872)",
            "matrix(0.399908 -0.916555 0.916555 0.399908 1454.44 1895)",
        ),
        "ET Direct": ("matrix(0.775633 -0.631184 0.631184 0.775633 1547.29 2089)", None),
        "PRE-SM-VOD Indirect": (
            "matrix(0.17392 -0.98476 0.98476 0.17392 1215.38 1995)",
            None,
        ),
        "ET-SM-VOD Indirect": (
            "matrix(0.809484 -0.587142 0.587142 0.809484 1650.49 2491)",
            None,
        ),
        "VPD-SM-VOD Indirect": (
            "matrix(0.641838 0.76684 -0.76684 0.641838 450.703 1679)",
            "matrix(0.641838 0.76684 -0.76684 0.641838 434.891 1660)",
        ),
        "VPD-ET-VOD Indirect": (
            "matrix(0.681625 0.731701 -0.731701 0.681625 377.694 2241)",
            None,
        ),
    },
    "nonforest": {
        "SM Direct": ("matrix(0.296933 0.954898 -0.954898 0.296933 3070.17 1642)", None),
        "VPD Direct": (
            "matrix(0.79894 0.60141 -0.60141 0.79894 2901.61 2159)",
            "matrix(0.79894 0.60141 -0.60141 0.79894 2881.93 2144)",
        ),
        "PRE Direct": (
            "matrix(0.417282 -0.908777 0.908777 0.417282 3643.97 1884)",
            "matrix(0.417282 -0.908777 0.908777 0.417282 3633.69 1906)",
        ),
        "ET Direct": ("matrix(0.775014 -0.631944 0.631944 0.775014 3724.46 2096)", None),
        "PRE-SM-VOD Indirect": (
            "matrix(0.17392 -0.98476 0.98476 0.17392 3393.43 2002)",
            None,
        ),
        "ET-SM-VOD Indirect": (
            "matrix(0.809484 -0.587142 0.587142 0.809484 3827.24 2497)",
            None,
        ),
        "VPD-SM-VOD Indirect": (
            "matrix(0.641838 0.76684 -0.76684 0.641838 2630.95 1687)",
            "matrix(0.641838 0.76684 -0.76684 0.641838 2615.14 1668)",
        ),
        "VPD-ET-VOD Indirect": (
            "matrix(0.739914 0.672702 -0.672702 0.739914 2569.78 2274)",
            "matrix(0.739914 0.672702 -0.672702 0.739914 2551.55 2257)",
        ),
    },
}


# Arrow paths are identified by stable attributes from the editable template.
# Only fill colour is changed; the original path geometry is preserved.
ARROW_SLOTS = {
    "forest": {
        "SM Direct": ("d", "M816.616 1446.2"),
        "VPD Direct": ("d", "M152.115 1745.49"),
        "PRE Direct": ("transform", "matrix(-1 0 0 1 1552.24 1456.5)"),
        "ET Direct": ("transform", "matrix(-1 0 0 1 1989.7 1759.5)"),
        "PRE-SM-VOD Indirect": (
            "transform",
            "matrix(-1 1.22465e-16 1.22465e-16 1 1395.21 1371.5)",
        ),
        "ET-SM-VOD Indirect": ("d", "M1990.32 1759.02"),
        "VPD-SM-VOD Indirect": ("d", "M305.694 1661.92"),
        "VPD-ET-VOD Indirect": (
            "transform",
            "matrix(1.83697e-16 1 1 -1.83697e-16 148.5 1751.5)",
        ),
    },
    "nonforest": {
        "SM Direct": ("d", "M2997.26 1454.7"),
        "VPD Direct": ("d", "M2332.24 1755.83"),
        "PRE Direct": ("transform", "matrix(-1 0 0 1 3729.9 1465.5)"),
        "ET Direct": ("transform", "matrix(-1 0 0 1 4165.52 1767)"),
        "PRE-SM-VOD Indirect": (
            "transform",
            "matrix(-1 1.22465e-16 1.22465e-16 1 3573.34 1380.5)",
        ),
        "ET-SM-VOD Indirect": ("d", "M4172.99 1767.11"),
        "VPD-SM-VOD Indirect": ("d", "M2486.86 1665.91"),
        "VPD-ET-VOD Indirect": (
            "transform",
            "matrix(1.83697e-16 1 1 -1.83697e-16 2329 1759)",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--forest", type=Path, default=DEFAULT_FOREST)
    parser.add_argument("--nonforest", type=Path, default=DEFAULT_NONFOREST)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        "--archive-template",
        type=Path,
        help="Copy the supplied template to this stable archive path before building.",
    )
    parser.add_argument("--skip-png", action="store_true")
    return parser.parse_args()


def mean_values(path: Path) -> dict[str, float]:
    columns = list(TEXT_SLOTS["forest"]) + [SERIAL_EFFECT]
    data = pd.read_csv(path, usecols=columns)
    values = data.mean(numeric_only=True).to_dict()
    if set(values) != set(columns):
        missing = sorted(set(columns) - set(values))
        raise RuntimeError(f"Missing numeric Figure 11 columns in {path}: {missing}")
    return {name: float(value) for name, value in values.items()}


def svg_elements(root: ET.Element, tag: str) -> list[ET.Element]:
    return list(root.iter(f"{{{SVG_NS}}}{tag}"))


def unique_by_attribute(
    elements: list[ET.Element], attribute: str, needle: str, *, prefix: bool = False
) -> ET.Element:
    if prefix:
        matches = [item for item in elements if item.get(attribute, "").startswith(needle)]
    else:
        matches = [item for item in elements if item.get(attribute) == needle]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one SVG element for {attribute}={needle!r}, found {len(matches)}"
        )
    return matches[0]


def update_text(root: ET.Element, group: str, values: dict[str, float]) -> None:
    texts = svg_elements(root, "text")
    for column, (value_transform, sign_transform) in TEXT_SLOTS[group].items():
        value = values[column]
        value_element = unique_by_attribute(texts, "transform", value_transform)
        if sign_transform is None:
            value_element.text = f"{value:.4f}"
        else:
            value_element.text = f"{abs(value):.4f}"
            sign_element = unique_by_attribute(texts, "transform", sign_transform)
            sign_element.text = "-" if value < 0 else ""


def update_arrows(root: ET.Element, group: str, values: dict[str, float]) -> None:
    paths = svg_elements(root, "path")
    for column, (attribute, needle) in ARROW_SLOTS[group].items():
        arrow = unique_by_attribute(paths, attribute, needle, prefix=attribute == "d")
        arrow.set("fill", POSITIVE if values[column] >= 0 else NEGATIVE)


def enlarge_panel_text(root: ET.Element) -> None:
    """Keep both panels equally legible after widening the outer margins."""

    for element in svg_elements(root, "text"):
        transform = element.get("transform", "")
        numbers = [
            float(value)
            for value in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", transform, re.IGNORECASE)
        ]
        if len(numbers) < 2:
            continue
        x, y = numbers[-2:]
        size = element.get("font-size")
        if y >= 1200 and size in PANEL_C_FONT_SIZES:
            element.set("font-size", PANEL_C_FONT_SIZES[size])


def add_element(parent: ET.Element, tag: str, **attributes) -> ET.Element:
    return ET.SubElement(parent, f"{{{SVG_NS}}}{tag}", {
        name.replace("_", "-"): str(value) for name, value in attributes.items()
    })


def new_text(parent: ET.Element, x: float, y: float, text: str,
             size: int = 73, **attributes) -> ET.Element:
    element = add_element(parent, "text", x=x, y=y,
                          font_family="Times New Roman", font_size=size,
                          fill="#000000", **attributes)
    element.text = text
    return element


def add_serial_path(panel: ET.Element, start: tuple[float, float],
                    end: tuple[float, float], value: float, name: str) -> None:
    """Route the fifth mediated effect outside the original ET-only curve."""
    sx, sy = start
    ex, ey = end
    color = POSITIVE if value >= 0 else NEGATIVE
    group = add_element(panel, "g", id=f"{name}-serial-effect")
    dy = sy - 1666
    points = [(sx+55, sy+85.5), (sx+25, 2210+dy),
              (sx+250, 2600+dy), (ex-157, ey-28)]
    start_point, c1, c2, finish = points
    add_element(group, "path",
                d=(f"M {start_point[0]} {start_point[1]} "
                   f"C {c1[0]} {c1[1]} {c2[0]} {c2[1]} {finish[0]} {finish[1]}"),
                fill="none", stroke=color, stroke_width=10,
                stroke_dasharray="34 19", marker_end=f"url(#{name}-serial-tip)")
    # Place the mediator capsule on the compact curve, aligned to its tangent.
    t = 0.60
    weights = ((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)
    cx, cy = [sum(w*p[axis] for w,p in zip(weights, points)) for axis in (0, 1)]
    tangent = [3*(1-t)**2*(c1[i]-start_point[i]) +
               6*(1-t)*t*(c2[i]-c1[i]) + 3*t*t*(finish[i]-c2[i]) for i in (0, 1)]
    angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    capsule = add_element(group, "g", transform=f"translate({cx} {cy}) rotate({angle})")
    add_element(capsule, "ellipse", cx=0, cy=0, rx=195, ry=71,
                fill="#FFFFFF", stroke="#000000", stroke_width=6)
    new_text(capsule, 0, 23, "ET \u2192 SM", size=72,
             text_anchor="middle", font_style="italic", font_weight="700")
    new_text(group, sx+730, 2700+dy, f"{value:.4f}", size=86,
             text_anchor="middle", font_weight="700")


def complete_effect_diagrams(root: ET.Element,
                            forest: dict[str, float],
                            nonforest: dict[str, float]) -> None:
    """Preserve template geometry and make space for two outer serial paths."""
    root.set("width", "4640")
    root.set("height", "2910")
    root.set("viewBox", "0 0 4640 2910")
    definitions = root.find(f"{{{SVG_NS}}}defs")
    main = root.find(f"{{{SVG_NS}}}g")
    children = list(main)
    # These ranges are stable in the archived, editable source template.
    assert len(children) == 88
    assert children[36].text == "b)" and children[85].text == "c)"
    assert children[38].text == "Direct " and children[46].text == "Negative path coefficient"
    for element in children:
        main.remove(element)
    for name, values in (("forest", forest), ("nonforest", nonforest)):
        marker = add_element(definitions, "marker", id=f"{name}-serial-tip",
                             markerUnits="userSpaceOnUse", markerWidth=48,
                             markerHeight=48, refX=48, refY=24, orient="auto",
                             overflow="visible")
        add_element(marker, "path", d="M 0 0 L 48 24 L 0 48 Z",
                    fill=POSITIVE if values[SERIAL_EFFECT] >= 0 else NEGATIVE,
                    stroke="none", stroke_dasharray="none")
    forest_panel = add_element(main, "g", id="forest-panel", transform="translate(50 0)")
    nonforest_panel = add_element(main, "g", id="nonforest-panel", transform="translate(200 0)")
    map_panel = add_element(main, "g", id="map-panel", transform="translate(202 0)")
    for element in children[:37]:
        forest_panel.append(element)
    for element in children[47:86]:
        nonforest_panel.append(element)
    # Normalize PowerPoint's nested crop transforms so narrowing the canvas
    # cannot clip the rightmost longitude label. Only the source title is cut.
    map_view = add_element(map_panel, "svg", x=547, y=100, width=3122,
                           height=1082, viewBox="0 120 3786 1312", overflow="hidden")
    source_image = definitions.find(f"{{{SVG_NS}}}image")
    image_attributes = {key: value for key, value in source_image.attrib.items() if key != "id"}
    ET.SubElement(map_view, f"{{{SVG_NS}}}image", image_attributes)
    map_panel.append(children[87])
    children[36].text = "b) Forest"
    children[85].text = "c) Non-forest"
    add_serial_path(forest_panel, (-8.5001, 1666), (1111.5, 2642.5),
                    forest[SERIAL_EFFECT], "forest")
    add_serial_path(nonforest_panel, (2173.5, 1674.5), (3290, 2647.5),
                    nonforest[SERIAL_EFFECT], "nonforest")

    legend = add_element(main, "g", id="shared-legend", transform="translate(-90 -210)")
    for x, label, dashed in ((450, "Direct effect", False), (1500, "Indirect effect", True)):
        attributes = {"stroke_dasharray": "30 16"} if dashed else {}
        add_element(legend, "path", d=f"M {x} 3060 H {x+130}",
                    stroke="#000000", stroke_width=12, fill="none", **attributes)
        add_element(legend, "path", d=f"M {x+130} 3038 L {x+178} 3060 L {x+130} 3082 Z",
                    fill="#000000")
        new_text(legend, x+205, 3083, label, size=70)
    for x, label, color in ((2450, "Positive coefficient", POSITIVE),
                            (3600, "Negative coefficient", NEGATIVE)):
        add_element(legend, "rect", x=x, y=3028, width=115, height=64,
                    stroke="#172C51", stroke_width=4, fill=color)
        new_text(legend, x+150, 3083, label, size=70)


def svg_dimensions(path: Path) -> tuple[str, str]:
    opening = path.read_text(encoding="utf-8", errors="ignore")[:1000]
    width = re.search(r'\bwidth="([0-9.]+)', opening)
    height = re.search(r'\bheight="([0-9.]+)', opening)
    if not width or not height:
        raise RuntimeError(f"SVG dimensions not found in {path}")
    return width.group(1), height.group(1)


def find_edge() -> Path:
    candidates = [
        Path(os.environ["FIGURE11_EDGE"]) if os.environ.get("FIGURE11_EDGE") else None,
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Microsoft Edge was not found. Set FIGURE11_EDGE or rerun with --skip-png."
    )


def png_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 20:
        return False
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return False
        handle.seek(-12, os.SEEK_END)
        return handle.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"


def export_png(svg: Path, png: Path) -> None:
    edge = find_edge()
    source_width, source_height = svg_dimensions(svg)
    width = int(round(float(source_width)))
    height = int(round(float(source_height)))
    png.parent.mkdir(parents=True, exist_ok=True)
    png.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="figure11_edge_") as profile:
        completed = subprocess.run(
            [
                str(edge),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--force-device-scale-factor=1",
                f"--window-size={width},{height}",
                f"--user-data-dir={profile}",
                f"--screenshot={png}",
                svg.resolve().as_uri(),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not png_is_complete(png):
            time.sleep(0.1)

    if not png_is_complete(png):
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Figure 11 PNG export failed: {detail}")


def main() -> None:
    args = parse_args()
    template = args.template.resolve()
    if not template.exists():
        raise FileNotFoundError(template)

    if args.archive_template:
        archive = args.archive_template.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive != template:
            shutil.copy2(template, archive)

    forest = mean_values(args.forest.resolve())
    nonforest = mean_values(args.nonforest.resolve())

    tree = ET.parse(template)
    root = tree.getroot()
    update_text(root, "forest", forest)
    update_text(root, "nonforest", nonforest)
    update_arrows(root, "forest", forest)
    update_arrows(root, "nonforest", nonforest)
    enlarge_panel_text(root)
    complete_effect_diagrams(root, forest, nonforest)

    args.svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.svg, encoding="utf-8", xml_declaration=False)
    # Keep an editable submission copy beside the standalone PNG.
    if args.svg.resolve() == DEFAULT_SVG.resolve():
        shutil.copy2(args.svg, BASE / "figures" / "Figure_11.svg")
    if not args.skip_png:
        export_png(args.svg.resolve(), args.png.resolve())

    print(f"Saved editable SVG: {args.svg.resolve()}")
    if not args.skip_png:
        print(f"Saved PNG: {args.png.resolve()}")
    print("Forest panel means:")
    for column in list(TEXT_SLOTS["forest"]) + [SERIAL_EFFECT]:
        print(f"  {column}: {forest[column]:.4f}")
    print(f"Non-forest serial effect: {nonforest[SERIAL_EFFECT]:.4f}")


if __name__ == "__main__":
    main()
