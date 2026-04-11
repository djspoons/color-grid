import numpy as np
from PIL import Image

from color_grid.grid import image_to_cell_colors
from color_grid.quantize import quantize_cells
from color_grid.render import render_page, render_solution


def _fixture_image() -> Image.Image:
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :20] = (255, 0, 0)
    arr[:20, 20:] = (0, 255, 0)
    arr[20:, :20] = (0, 0, 255)
    arr[20:, 20:] = (255, 255, 0)
    return Image.fromarray(arr)


def test_end_to_end():
    image = _fixture_image()
    cells = image_to_cell_colors(image, width=2, height=2)
    assert cells.shape == (2, 2, 3)

    labels, palette = quantize_cells(cells, n_colors=4)
    assert labels.shape == (2, 2)
    assert palette.shape == (4, 3)
    assert len(set(labels.flatten().tolist())) == 4

    page = render_page(labels, palette)
    assert page.size[0] > 0 and page.size[1] > 0

    solution = render_solution(labels, palette)
    assert solution.size[0] > 0 and solution.size[1] > 0


def test_fewer_unique_colors_than_requested():
    image = Image.new("RGB", (20, 20), (128, 64, 200))
    cells = image_to_cell_colors(image, width=2, height=2)
    labels, palette = quantize_cells(cells, n_colors=5)
    assert palette.shape[0] == 1
    assert (labels == 0).all()
