from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab


PAPER_SIZES_INCHES = {
    "letter": (8.5, 11.0),
    "legal": (8.5, 14.0),
    "a4": (8.27, 11.69),
    "a5": (5.83, 8.27),
}


@dataclass(frozen=True)
class PageSpec:
    paper: str
    margin_in: float = 0.5

    @property
    def size_pt(self) -> tuple[float, float]:
        w_in, h_in = PAPER_SIZES_INCHES[self.paper]
        return (w_in * 72.0, h_in * 72.0)

    @property
    def margin_pt(self) -> float:
        return self.margin_in * 72.0


def _legend_order(palette: np.ndarray) -> list[int]:
    """Order legend entries by nearest-neighbor walk in CIELAB space.

    Starts from the lightest color and greedily picks the closest unvisited
    color, so perceptually similar colors end up adjacent in the legend.
    """
    n = len(palette)
    if n <= 1:
        return list(range(n))

    rgb01 = palette.astype(np.float64) / 255.0
    lab = rgb2lab(rgb01.reshape(1, -1, 3)).reshape(-1, 3)

    # Start from the lightest color (highest L*)
    current = int(np.argmax(lab[:, 0]))
    visited = [current]
    remaining = set(range(n)) - {current}

    while remaining:
        dists = np.linalg.norm(lab[list(remaining)] - lab[current], axis=1)
        nearest = list(remaining)[int(np.argmin(dists))]
        visited.append(nearest)
        remaining.remove(nearest)
        current = nearest

    return visited


def _layout(page: PageSpec, grid_w: int, grid_h: int, n_colors: int) -> dict:
    """Compute cell size and legend geometry in points (1/72 inch)."""
    pw, ph = page.size_pt
    m = page.margin_pt
    avail_w = pw - 2 * m
    avail_h = ph - 2 * m

    swatch = 0.35 * 72.0
    legend_gap = 0.12 * 72.0
    legend_label_w = 0.55 * 72.0
    legend_col_w = swatch + legend_label_w
    legend_cols = max(1, int(avail_w // legend_col_w))
    legend_rows = (n_colors + legend_cols - 1) // legend_cols
    legend_h = legend_rows * swatch + (legend_rows - 1) * legend_gap
    grid_legend_gap = 0.3 * 72.0

    grid_area_h = avail_h - legend_h - grid_legend_gap
    if grid_area_h <= 0:
        raise ValueError(
            f"{n_colors} colors leave no room for the grid on {page.paper}; "
            "reduce colors or use larger paper."
        )

    cell_pt = min(avail_w / grid_w, grid_area_h / grid_h)
    if cell_pt < 10 * (72.0 / 300.0):
        raise ValueError(
            f"grid {grid_w}x{grid_h} is too large for {page.paper}"
        )

    grid_px_w = cell_pt * grid_w
    grid_px_h = cell_pt * grid_h
    grid_x = (pw - grid_px_w) / 2.0
    grid_y = m

    legend_y = grid_y + grid_px_h + grid_legend_gap
    legend_total_w = legend_cols * legend_col_w
    legend_x = (pw - legend_total_w) / 2.0

    return {
        "cell_pt": cell_pt,
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
    fmt: str = "pdf",
    line_width: float | None = None,
) -> bytes:
    """Render a printable color-by-number page.

    `entry_labels`, if provided, is a length-n list of strings — one per
    palette entry — used in both the grid cells and the legend. Defaults to
    "1".."n".

    `fmt` is "pdf" or "svg".

    `line_width`, if provided, sets the grid line width in points. Defaults to
    auto-scaling based on cell size.
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
    lay["line_width"] = line_width
    order = _legend_order(palette)

    if fmt == "svg":
        from ._backend_svg import render_page_svg

        return render_page_svg(labels, palette, lay, page, entry_labels, order)
    else:
        from ._backend_pdf import render_page_pdf

        return render_page_pdf(labels, palette, lay, page, entry_labels, order)


def render_solution(
    labels: np.ndarray,
    palette: np.ndarray,
    page: PageSpec,
    fmt: str = "pdf",
    line_width: float | None = None,
) -> bytes:
    """Render a filled-in preview on the same page spec."""
    h, w = labels.shape
    lay = _layout(page, w, h, len(palette))
    lay["line_width"] = line_width

    if fmt == "svg":
        from ._backend_svg import render_solution_svg

        return render_solution_svg(labels, palette, lay, page)
    else:
        from ._backend_pdf import render_solution_pdf

        return render_solution_pdf(labels, palette, lay, page)


def save_page(data: bytes, path: Path) -> None:
    """Write rendered page bytes to disk."""
    path.write_bytes(data)
