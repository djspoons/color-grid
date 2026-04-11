import numpy as np
from PIL import Image

from color_grid.grid import image_to_cell_colors
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

    labels, palette = quantize_cells(cells, n_colors=4)
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
    labels, palette = quantize_cells(cells, n_colors=5)
    assert palette.shape[0] == 1
    assert (labels == 0).all()


def test_a4_and_legal_sizes():
    labels = np.zeros((4, 4), dtype=int)
    palette = np.array([[10, 20, 30]], dtype=np.uint8)
    for paper in ("a4", "legal"):
        spec = PageSpec(paper=paper, dpi=150)
        img = render_page(labels, palette, spec)
        assert img.size == spec.size_px
