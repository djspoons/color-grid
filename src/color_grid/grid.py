import numpy as np
from PIL import Image


def image_to_cell_colors(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Average an image down into a (height, width, 3) array of RGB cell colors.

    The image is cropped to a multiple of the grid dimensions so each cell
    covers an equal-sized block of source pixels.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    rgb = image.convert("RGB")
    src_w, src_h = rgb.size
    if src_w < width or src_h < height:
        raise ValueError(
            f"image ({src_w}x{src_h}) is smaller than grid ({width}x{height})"
        )

    cell_w = src_w // width
    cell_h = src_h // height
    cropped = rgb.crop((0, 0, cell_w * width, cell_h * height))

    arr = np.asarray(cropped, dtype=np.float32)
    # Reshape to (height, cell_h, width, cell_w, 3) then mean over cell axes.
    blocks = arr.reshape(height, cell_h, width, cell_w, 3)
    cells = blocks.mean(axis=(1, 3))
    return cells
