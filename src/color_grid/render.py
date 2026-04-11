import numpy as np
from PIL import Image, ImageDraw, ImageFont


CELL_PX = 60
BORDER_PX = 2
MARGIN_PX = 40
LEGEND_SWATCH = 40
LEGEND_GAP = 12


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_page(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Render a color-by-number page: numbered grid on top, legend below."""
    h, w = labels.shape
    n_colors = len(palette)

    grid_w = w * CELL_PX + BORDER_PX
    grid_h = h * CELL_PX + BORDER_PX

    legend_cols = max(1, (grid_w - 2 * MARGIN_PX) // (LEGEND_SWATCH + 80))
    legend_rows = (n_colors + legend_cols - 1) // legend_cols
    legend_h = legend_rows * (LEGEND_SWATCH + LEGEND_GAP)

    page_w = grid_w + 2 * MARGIN_PX
    page_h = grid_h + legend_h + 3 * MARGIN_PX

    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)

    number_font = _load_font(CELL_PX // 2)
    legend_font = _load_font(LEGEND_SWATCH // 2)

    origin_x = MARGIN_PX
    origin_y = MARGIN_PX

    for row in range(h):
        for col in range(w):
            x0 = origin_x + col * CELL_PX
            y0 = origin_y + row * CELL_PX
            x1 = x0 + CELL_PX
            y1 = y0 + CELL_PX
            draw.rectangle([x0, y0, x1, y1], outline="black", width=BORDER_PX)

            label = int(labels[row, col]) + 1
            text = str(label)
            bbox = draw.textbbox((0, 0), text, font=number_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (CELL_PX - tw) // 2 - bbox[0]
            ty = y0 + (CELL_PX - th) // 2 - bbox[1]
            draw.text((tx, ty), text, fill="black", font=number_font)

    legend_y = origin_y + grid_h + MARGIN_PX
    col_stride = (page_w - 2 * MARGIN_PX) // legend_cols
    for i, color in enumerate(palette):
        col = i % legend_cols
        row = i // legend_cols
        x0 = MARGIN_PX + col * col_stride
        y0 = legend_y + row * (LEGEND_SWATCH + LEGEND_GAP)
        draw.rectangle(
            [x0, y0, x0 + LEGEND_SWATCH, y0 + LEGEND_SWATCH],
            fill=tuple(int(c) for c in color),
            outline="black",
            width=BORDER_PX,
        )
        draw.text(
            (x0 + LEGEND_SWATCH + 8, y0 + LEGEND_SWATCH // 4),
            str(i + 1),
            fill="black",
            font=legend_font,
        )

    return page


def render_solution(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Render the filled-in grid (preview of the finished result)."""
    h, w = labels.shape
    grid_w = w * CELL_PX + BORDER_PX
    grid_h = h * CELL_PX + BORDER_PX

    page = Image.new("RGB", (grid_w + 2 * MARGIN_PX, grid_h + 2 * MARGIN_PX), "white")
    draw = ImageDraw.Draw(page)

    for row in range(h):
        for col in range(w):
            x0 = MARGIN_PX + col * CELL_PX
            y0 = MARGIN_PX + row * CELL_PX
            color = tuple(int(c) for c in palette[int(labels[row, col])])
            draw.rectangle(
                [x0, y0, x0 + CELL_PX, y0 + CELL_PX],
                fill=color,
                outline="black",
                width=BORDER_PX,
            )

    return page
