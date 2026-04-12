"""Vector SVG rendering — hand-written XML, no external dependency."""

from xml.sax.saxutils import escape

import numpy as np


def _svg_color(rgb: np.ndarray) -> str:
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def render_page_svg(
    labels: np.ndarray,
    palette: np.ndarray,
    lay: dict,
    page,
    keys: list[str],
    codes: list[str] | None,
    order: list[int],
) -> bytes:
    """Render a color-by-number page as SVG. Returns UTF-8 bytes."""
    h, w = labels.shape
    pw, ph = page.size_pt
    cell = lay["cell_pt"]
    border = lay["line_width"] if lay.get("line_width") is not None else max(0.5, cell * 0.04)
    gx, gy = lay["grid_x"], lay["grid_y"]

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{pw:.2f}pt" height="{ph:.2f}pt" '
        f'viewBox="0 0 {pw:.2f} {ph:.2f}">'
    )
    parts.append(
        '<rect width="100%" height="100%" fill="white"/>'
    )

    # --- Grid ---
    font_size = cell * 0.65
    font_attrs = (
        'font-family="Helvetica, Arial, sans-serif" '
        'font-weight="normal" '
        'text-anchor="middle" '
        'dominant-baseline="central"'
    )

    for row in range(h):
        for col in range(w):
            x0 = gx + col * cell
            y0 = gy + row * cell
            parts.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" '
                f'width="{cell:.2f}" height="{cell:.2f}" '
                f'fill="white" stroke="black" stroke-width="{border:.2f}"/>'
            )
            text = escape(keys[int(labels[row, col])])
            cx = x0 + cell / 2.0
            cy = y0 + cell / 2.0
            parts.append(
                f'<text x="{cx:.2f}" y="{cy:.2f}" '
                f'font-size="{font_size:.2f}" {font_attrs}>'
                f'{text}</text>'
            )

    # --- Legend ---
    swatch = lay["swatch"]
    legend_font_size = max(8.0, swatch / 2.0)
    key_col_w = swatch * 0.6

    for slot, i in enumerate(order):
        color = palette[i]
        col_idx = slot % lay["legend_cols"]
        row_idx = slot // lay["legend_cols"]
        x0 = lay["legend_x"] + col_idx * lay["legend_col_w"]
        y0 = lay["legend_y"] + row_idx * (swatch + lay["legend_gap"])

        # swatch
        parts.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" '
            f'width="{swatch:.2f}" height="{swatch:.2f}" '
            f'fill="{_svg_color(color)}" stroke="black" '
            f'stroke-width="{border:.2f}"/>'
        )

        # key
        kx = x0 + swatch + swatch / 4.0
        ky = y0 + swatch / 2.0
        parts.append(
            f'<text x="{kx:.2f}" y="{ky:.2f}" '
            f'font-size="{legend_font_size:.2f}" '
            f'font-family="Helvetica, Arial, sans-serif" '
            f'dominant-baseline="central">'
            f'{escape(keys[i])}</text>'
        )

        # code (if available)
        if codes is not None:
            cx = kx + key_col_w
            code_font_size = legend_font_size * 0.85
            parts.append(
                f'<text x="{cx:.2f}" y="{ky:.2f}" '
                f'font-size="{code_font_size:.2f}" '
                f'font-family="Helvetica, Arial, sans-serif" '
                f'dominant-baseline="central" fill="#666">'
                f'{escape(codes[i])}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts).encode("utf-8")


def render_solution_svg(
    labels: np.ndarray,
    palette: np.ndarray,
    lay: dict,
    page,
) -> bytes:
    """Render a filled-in solution preview as SVG."""
    h, w = labels.shape
    pw, ph = page.size_pt
    cell = lay["cell_pt"]
    border = lay["line_width"] if lay.get("line_width") is not None else max(0.5, cell * 0.04)
    gx, gy = lay["grid_x"], lay["grid_y"]

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{pw:.2f}pt" height="{ph:.2f}pt" '
        f'viewBox="0 0 {pw:.2f} {ph:.2f}">'
    )
    parts.append(
        '<rect width="100%" height="100%" fill="white"/>'
    )

    for row in range(h):
        for col in range(w):
            x0 = gx + col * cell
            y0 = gy + row * cell
            color = palette[int(labels[row, col])]
            parts.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" '
                f'width="{cell:.2f}" height="{cell:.2f}" '
                f'fill="{_svg_color(color)}" stroke="black" '
                f'stroke-width="{border:.2f}"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts).encode("utf-8")
