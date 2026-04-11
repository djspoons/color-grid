import colorsys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PAPER_SIZES_INCHES = {
    "letter": (8.5, 11.0),
    "legal": (8.5, 14.0),
    "a4": (8.27, 11.69),
    "a5": (5.83, 8.27),
}


@dataclass(frozen=True)
class PageSpec:
    paper: str
    dpi: int
    margin_in: float = 0.5

    @property
    def size_px(self) -> tuple[int, int]:
        w_in, h_in = PAPER_SIZES_INCHES[self.paper]
        return (round(w_in * self.dpi), round(h_in * self.dpi))

    @property
    def margin_px(self) -> int:
        return round(self.margin_in * self.dpi)


def _legend_order(palette: np.ndarray) -> list[int]:
    """Order legend entries so colorers can find a swatch quickly.

    Low-saturation entries (grays) are grouped first, light-to-dark, so they
    don't land randomly among the chromatic colors. The rest are sorted by
    hue, then by saturation and value as tiebreakers.
    """
    keys: list[tuple] = []
    for i, rgb in enumerate(palette):
        r, g, b = [float(c) / 255.0 for c in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s < 0.15:
            keys.append((0, -v, 0.0, 0.0, i))
        else:
            keys.append((1, h, -s, -v, i))
    keys.sort()
    return [k[-1] for k in keys]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _layout(page: PageSpec, grid_w: int, grid_h: int, n_colors: int) -> dict:
    """Compute cell size and legend geometry that fits on the page."""
    pw, ph = page.size_px
    m = page.margin_px
    avail_w = pw - 2 * m
    avail_h = ph - 2 * m

    swatch = round(0.35 * page.dpi)
    legend_gap = round(0.12 * page.dpi)
    legend_label_w = round(0.55 * page.dpi)
    legend_col_w = swatch + legend_label_w
    legend_cols = max(1, avail_w // legend_col_w)
    legend_rows = (n_colors + legend_cols - 1) // legend_cols
    legend_h = legend_rows * swatch + (legend_rows - 1) * legend_gap
    grid_legend_gap = round(0.3 * page.dpi)

    grid_area_h = avail_h - legend_h - grid_legend_gap
    if grid_area_h <= 0:
        raise ValueError(
            f"{n_colors} colors leave no room for the grid on {page.paper}; "
            "reduce colors or use larger paper."
        )

    cell_px = min(avail_w // grid_w, grid_area_h // grid_h)
    if cell_px < 10:
        raise ValueError(
            f"grid {grid_w}x{grid_h} is too large for {page.paper} at {page.dpi} dpi"
        )

    grid_px_w = cell_px * grid_w
    grid_px_h = cell_px * grid_h
    grid_x = (pw - grid_px_w) // 2
    grid_y = m

    legend_y = grid_y + grid_px_h + grid_legend_gap
    legend_total_w = legend_cols * legend_col_w
    legend_x = (pw - legend_total_w) // 2

    return {
        "cell_px": cell_px,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "swatch": swatch,
        "legend_gap": legend_gap,
        "legend_cols": legend_cols,
        "legend_col_w": legend_col_w,
        "legend_x": legend_x,
        "legend_y": legend_y,
    }


def render_page(
    labels: np.ndarray,
    palette: np.ndarray,
    page: PageSpec,
    entry_labels: list[str] | None = None,
) -> Image.Image:
    """Render a printable color-by-number page.

    `entry_labels`, if provided, is a length-n list of strings — one per
    palette entry — used in both the grid cells and the legend. Defaults to
    "1".."n".
    """
    h, w = labels.shape
    n_colors = len(palette)
    if entry_labels is None:
        entry_labels = [str(i + 1) for i in range(n_colors)]
    elif len(entry_labels) != n_colors:
        raise ValueError(
            f"entry_labels has {len(entry_labels)} items but palette has {n_colors}"
        )
    lay = _layout(page, w, h, n_colors)

    cell = lay["cell_px"]
    border = max(1, round(cell * 0.04))

    img = Image.new("RGB", page.size_px, "white")
    draw = ImageDraw.Draw(img)

    longest = max(len(t) for t in entry_labels) or 1
    number_font = _load_font(max(10, int(cell * 0.7 / longest)))

    gx, gy = lay["grid_x"], lay["grid_y"]
    for row in range(h):
        for col in range(w):
            x0 = gx + col * cell
            y0 = gy + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], outline="black", width=border)

            text = entry_labels[int(labels[row, col])]
            bbox = draw.textbbox((0, 0), text, font=number_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell - tw) // 2 - bbox[0]
            ty = y0 + (cell - th) // 2 - bbox[1]
            draw.text((tx, ty), text, fill="black", font=number_font)

    swatch = lay["swatch"]
    legend_font = _load_font(max(10, swatch // 2))
    order = _legend_order(palette)
    for slot, i in enumerate(order):
        color = palette[i]
        col = slot % lay["legend_cols"]
        row = slot // lay["legend_cols"]
        x0 = lay["legend_x"] + col * lay["legend_col_w"]
        y0 = lay["legend_y"] + row * (swatch + lay["legend_gap"])
        draw.rectangle(
            [x0, y0, x0 + swatch, y0 + swatch],
            fill=tuple(int(c) for c in color),
            outline="black",
            width=border,
        )
        draw.text(
            (x0 + swatch + swatch // 4, y0 + swatch // 4),
            entry_labels[i],
            fill="black",
            font=legend_font,
        )

    return img


def render_solution(labels: np.ndarray, palette: np.ndarray, page: PageSpec) -> Image.Image:
    """Render a filled-in preview on the same page spec."""
    h, w = labels.shape
    lay = _layout(page, w, h, len(palette))
    cell = lay["cell_px"]
    border = max(1, round(cell * 0.04))

    img = Image.new("RGB", page.size_px, "white")
    draw = ImageDraw.Draw(img)

    gx, gy = lay["grid_x"], lay["grid_y"]
    for row in range(h):
        for col in range(w):
            x0 = gx + col * cell
            y0 = gy + row * cell
            color = tuple(int(c) for c in palette[int(labels[row, col])])
            draw.rectangle(
                [x0, y0, x0 + cell, y0 + cell],
                fill=color,
                outline="black",
                width=border,
            )
    return img


def save_page(image: Image.Image, path, page: PageSpec) -> None:
    """Save as PDF (default) or PNG based on the path suffix."""
    suffix = path.suffix.lower()
    if suffix == ".pdf" or suffix == "":
        if suffix == "":
            path = path.with_suffix(".pdf")
        image.save(path, "PDF", resolution=page.dpi)
    else:
        image.save(path, dpi=(page.dpi, page.dpi))
