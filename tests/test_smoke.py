import numpy as np
from PIL import Image

from pathlib import Path

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

    page_spec = PageSpec(paper="letter", dpi=150)
    page = render_page(labels, palette, page_spec)
    assert page.size == page_spec.size_px

    out_pdf = tmp_path / "out.pdf"
    save_page(page, out_pdf, page_spec)
    assert out_pdf.exists() and out_pdf.stat().st_size > 0

    out_png = tmp_path / "out.png"
    save_page(page, out_png, page_spec)
    assert out_png.exists()

    solution = render_solution(labels, palette, page_spec)
    assert solution.size == page_spec.size_px


def test_fewer_unique_colors_than_requested():
    image = Image.new("RGB", (20, 20), (128, 64, 200))
    cells = image_to_cell_colors(image, width=2, height=2)
    labels, palette, chosen = quantize_cells(cells, n_colors=5)
    assert palette.shape[0] == 1
    assert (labels == 0).all()
    assert chosen is None


def test_maxcoverage_preserves_rare_vivid_color():
    # Three distinct brown modes dominate the image, leaving k-means with no
    # budget for a single-cell red. maxcoverage should still pick it up.
    cells = np.zeros((30, 30, 3), dtype=np.float32)
    cells[:10, :] = (120, 80, 40)
    cells[10:20, :] = (150, 100, 60)
    cells[20:, :] = (90, 60, 30)
    cells[15, 15] = (230, 20, 20)  # single vivid red cell

    def dist_to_red(palette):
        red = np.array([230, 20, 20])
        return np.min(np.linalg.norm(palette.astype(float) - red, axis=1))

    _, km_palette, _ = quantize_cells(cells, n_colors=3, method="kmeans")
    _, mc_palette, _ = quantize_cells(cells, n_colors=3, method="maxcoverage")

    assert dist_to_red(km_palette) > 100  # k-means ignored the red
    assert dist_to_red(mc_palette) < 20   # maxcoverage landed on it


def test_fixed_palette_snaps_to_palette_entries():
    cells = np.zeros((4, 4, 3), dtype=np.float32)
    cells[:2, :2] = (250, 10, 10)
    cells[:2, 2:] = (10, 250, 10)
    cells[2:, :2] = (10, 10, 250)
    cells[2:, 2:] = (240, 240, 20)

    # Palette close to but not exactly the cell colors, plus noise entries.
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
    # Every output color must be an exact entry from the fixed palette.
    fixed_set = {tuple(c) for c in fixed.tolist()}
    for color in out_palette.tolist():
        assert tuple(color) in fixed_set
    # And the four distinct cells should pick the four pure primaries.
    expected = {(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)}
    assert {tuple(c) for c in out_palette.tolist()} == expected




def test_load_palette_srgb_format(tmp_path):
    import json as _json
    p = tmp_path / "pal.json"
    p.write_text(_json.dumps({
        "name": "Test",
        "color_space": "srgb",
        "colors": [
            {"code": "A", "color": [255, 0, 0]},
            {"code": "B", "color": [0, 255, 0]},
        ],
    }))
    pal = load_palette(p)
    assert pal.codes == ["A", "B"]
    assert pal.rgb.tolist() == [[255, 0, 0], [0, 255, 0]]


def test_load_palette_lab_format_roundtrips_primaries(tmp_path):
    import json as _json
    # LAB of pure red in D65: ~(53.24, 80.09, 67.20)
    p = tmp_path / "pal.json"
    p.write_text(_json.dumps({
        "name": "Test",
        "color_space": "lab",
        "colors": [{"code": "R", "color": [53.24, 80.09, 67.20]}],
    }))
    pal = load_palette(p)
    r, g, b = pal.rgb[0]
    assert r > 240 and g < 15 and b < 15


def test_load_palette_rejects_unknown_space(tmp_path):
    import json as _json
    p = tmp_path / "pal.json"
    p.write_text(_json.dumps({"name": "t", "color_space": "xyz", "colors": [
        {"code": "1", "color": [0, 0, 0]}
    ]}))
    import pytest
    with pytest.raises(ValueError):
        load_palette(p)


def test_legend_order_sorts_by_hue():
    from color_grid.render import _legend_order
    palette = np.array(
        [
            [200, 200, 200],  # light gray
            [30, 30, 30],     # dark gray
            [200, 30, 30],    # red
            [30, 200, 30],    # green
            [30, 30, 200],    # blue
        ],
        dtype=np.uint8,
    )
    order = _legend_order(palette)
    # Grays come first, light before dark.
    assert order[:2] == [0, 1]
    # Chromatic entries follow, sorted by hue (red < green < blue in HSV).
    assert order[2:] == [2, 3, 4]


def test_render_page_uses_entry_labels(tmp_path):
    labels = np.array([[0, 1], [1, 0]], dtype=int)
    palette = np.array([[200, 30, 30], [30, 200, 30]], dtype=np.uint8)
    page_spec = PageSpec(paper="letter", dpi=100)
    img = render_page(labels, palette, page_spec, entry_labels=["R3", "G5"])
    assert img.size == page_spec.size_px

    import pytest
    with pytest.raises(ValueError):
        render_page(labels, palette, page_spec, entry_labels=["only-one"])


def test_a4_and_legal_sizes():
    labels = np.zeros((4, 4), dtype=int)
    palette = np.array([[10, 20, 30]], dtype=np.uint8)
    for paper in ("a4", "legal"):
        spec = PageSpec(paper=paper, dpi=150)
        img = render_page(labels, palette, spec)
        assert img.size == spec.size_px
