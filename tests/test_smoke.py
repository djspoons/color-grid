import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from color_grid.grid import image_to_cell_colors
from color_grid.palette import load_palette
from color_grid.quantize import quantize_cells
from color_grid.render import PageSpec, render_page, render_solution, save_page


def _fixture_image() -> Image.Image:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :20] = (255, 0, 0)
    arr[:20, 20:] = (0, 255, 0)
    arr[20:, :20] = (0, 0, 255)
    arr[20:, 20:] = (255, 255, 0)
    return Image.fromarray(arr)


def test_end_to_end(tmp_path):
    image = _fixture_image()
    cells = image_to_cell_colors(image, width=2, height=2)
    assert cells.shape == (2, 2, 3)

    for space in ("rgb", "lab", "ciecam16"):
        for method in ("kmeans", "maxcoverage"):
            labels, palette, chosen = quantize_cells(
                cells, n_colors=4, color_space=space, method=method
            )
            assert chosen is None
            assert labels.shape == (2, 2)
            assert palette.shape == (4, 3)
            assert len(set(labels.flatten().tolist())) == 4

    page_spec = PageSpec(paper="letter")

    # PDF output
    pdf_data = render_page(labels, palette, page_spec)
    assert isinstance(pdf_data, bytes) and len(pdf_data) > 0
    assert pdf_data[:5] == b"%PDF-"

    out_pdf = tmp_path / "out.pdf"
    save_page(pdf_data, out_pdf)
    assert out_pdf.exists() and out_pdf.stat().st_size > 0

    # SVG output
    svg_data = render_page(labels, palette, page_spec, fmt="svg")
    assert isinstance(svg_data, bytes) and len(svg_data) > 0
    ET.fromstring(svg_data)  # valid XML

    out_svg = tmp_path / "out.svg"
    save_page(svg_data, out_svg)
    assert out_svg.exists() and out_svg.stat().st_size > 0

    # Solution
    sol_data = render_solution(labels, palette, page_spec)
    assert isinstance(sol_data, bytes) and sol_data[:5] == b"%PDF-"

    sol_svg = render_solution(labels, palette, page_spec, fmt="svg")
    ET.fromstring(sol_svg)


def test_fewer_unique_colors_than_requested():
    image = Image.new("RGB", (20, 20), (128, 64, 200))
    cells = image_to_cell_colors(image, width=2, height=2)
    labels, palette, chosen = quantize_cells(cells, n_colors=5)
    assert palette.shape[0] == 1
    assert (labels == 0).all()
    assert chosen is None


def test_maxcoverage_preserves_rare_vivid_color():
    cells = np.zeros((30, 30, 3), dtype=np.float32)
    cells[:10, :] = (120, 80, 40)
    cells[10:20, :] = (150, 100, 60)
    cells[20:, :] = (90, 60, 30)
    cells[15, 15] = (230, 20, 20)

    def dist_to_red(palette):
        red = np.array([230, 20, 20])
        return np.min(np.linalg.norm(palette.astype(float) - red, axis=1))

    _, km_palette, _ = quantize_cells(cells, n_colors=3, method="kmeans")
    _, mc_palette, _ = quantize_cells(cells, n_colors=3, method="maxcoverage")

    assert dist_to_red(km_palette) > 100
    assert dist_to_red(mc_palette) < 20


def test_fixed_palette_snaps_to_palette_entries():
    cells = np.zeros((4, 4, 3), dtype=np.float32)
    cells[:2, :2] = (250, 10, 10)
    cells[:2, 2:] = (10, 250, 10)
    cells[2:, :2] = (10, 10, 250)
    cells[2:, 2:] = (240, 240, 20)

    fixed = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [120, 120, 120],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _, out_palette, chosen = quantize_cells(cells, n_colors=4, fixed_palette=fixed)
    assert chosen is not None and chosen.shape == (4,)
    fixed_set = {tuple(c) for c in fixed.tolist()}
    for color in out_palette.tolist():
        assert tuple(color) in fixed_set
    expected = {(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)}
    assert {tuple(c) for c in out_palette.tolist()} == expected


def test_load_palette_srgb_format(tmp_path):
    import json as _json
    p = tmp_path / "pal.json"
    p.write_text(_json.dumps({
        "name": "Test",
        "color_space": "srgb",
        "colors": [
            {"name": "A", "color": [255, 0, 0]},
            {"name": "B", "color": [0, 255, 0]},
        ],
    }))
    pal = load_palette(p)
    assert pal.color_names == ["A", "B"]
    assert pal.rgb.tolist() == [[255, 0, 0], [0, 255, 0]]


def test_load_palette_lab_format_roundtrips_primaries(tmp_path):
    import json as _json
    p = tmp_path / "pal.json"
    p.write_text(_json.dumps({
        "name": "Test",
        "color_space": "lab",
        "colors": [{"name": "R", "color": [53.24, 80.09, 67.20]}],
    }))
    pal = load_palette(p)
    r, g, b = pal.rgb[0]
    assert r > 240 and g < 15 and b < 15


def test_load_palette_rejects_unknown_space(tmp_path):
    import json as _json
    p = tmp_path / "pal.json"
    p.write_text(_json.dumps({"name": "t", "color_space": "xyz", "colors": [
        {"name": "1", "color": [0, 0, 0]}
    ]}))
    with pytest.raises(ValueError):
        load_palette(p)


def test_legend_order_groups_similar_colors():
    from color_grid.render import _legend_order
    palette = np.array(
        [
            [200, 30, 30],    # 0: red
            [220, 50, 50],    # 1: similar red
            [30, 30, 200],    # 2: blue
            [50, 50, 220],    # 3: similar blue
        ],
        dtype=np.uint8,
    )
    order = _legend_order(palette)
    # The two reds should be adjacent, and the two blues should be adjacent.
    red_positions = {order.index(0), order.index(1)}
    blue_positions = {order.index(2), order.index(3)}
    assert max(red_positions) - min(red_positions) == 1
    assert max(blue_positions) - min(blue_positions) == 1


def test_render_page_keys_and_codes():
    labels = np.array([[0, 1], [1, 0]], dtype=int)
    palette = np.array([[200, 30, 30], [30, 200, 30]], dtype=np.uint8)
    page_spec = PageSpec(paper="letter")

    # Auto-assigned keys (no explicit keys)
    data = render_page(labels, palette, page_spec)
    assert isinstance(data, bytes) and len(data) > 0

    # Explicit keys + codes
    data = render_page(labels, palette, page_spec, keys=["R", "G"], codes=["Red", "Green"])
    assert isinstance(data, bytes) and len(data) > 0

    # Wrong number of keys
    with pytest.raises(ValueError):
        render_page(labels, palette, page_spec, keys=["X"])


def test_a4_and_legal_sizes():
    labels = np.zeros((4, 4), dtype=int)
    palette = np.array([[10, 20, 30]], dtype=np.uint8)
    for paper in ("a4", "legal"):
        spec = PageSpec(paper=paper)
        data = render_page(labels, palette, spec)
        assert isinstance(data, bytes) and len(data) > 0
        assert data[:5] == b"%PDF-"


def test_svg_output_is_valid_xml():
    labels = np.array([[0, 1], [1, 0]], dtype=int)
    palette = np.array([[200, 30, 30], [30, 200, 30]], dtype=np.uint8)
    page_spec = PageSpec(paper="letter")

    svg_data = render_page(labels, palette, page_spec, fmt="svg")
    root = ET.fromstring(svg_data)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"

    sol_data = render_solution(labels, palette, page_spec, fmt="svg")
    root = ET.fromstring(sol_data)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
