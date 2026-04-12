"""Vector PDF rendering via reportlab."""

import io

import numpy as np
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


def _fit_font_size(text: str, max_w: float, max_h: float, font_name: str) -> float:
    """Binary-search for the largest font size where `text` fits in max_w x max_h."""
    face = pdfmetrics.getFont(font_name).face
    lo, hi = 4.0, max_h * 1.2
    best = lo
    while hi - lo > 0.5:
        mid = (lo + hi) / 2.0
        tw = pdfmetrics.stringWidth(text, font_name, mid)
        # Use ascent + descent from font metrics for height
        th = (face.ascent - face.descent) / 1000.0 * mid
        if tw <= max_w and th <= max_h:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def render_page_pdf(
    labels: np.ndarray,
    palette: np.ndarray,
    lay: dict,
    page,
    entry_labels: list[str],
    order: list[int],
) -> bytes:
    """Render a color-by-number page as a vector PDF. Returns PDF bytes."""
    h, w = labels.shape
    pw, ph = page.size_pt
    cell = lay["cell_pt"]
    border = lay["line_width"] if lay.get("line_width") is not None else max(0.5, cell * 0.04)
    gx, gy_top = lay["grid_x"], lay["grid_y"]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(pw, ph))

    # --- Grid ---
    longest_text = max(entry_labels, key=len)
    inner = cell * 0.78
    font_name = "Helvetica"
    font_size = _fit_font_size(longest_text, inner, inner, font_name)
    c.setFont(font_name, font_size)
    c.setLineWidth(border)

    face = pdfmetrics.getFont(font_name).face
    ascent_pt = face.ascent / 1000.0 * font_size
    descent_pt = face.descent / 1000.0 * font_size
    text_h = ascent_pt - descent_pt

    for row in range(h):
        for col in range(w):
            x0 = gx + col * cell
            y0_top = gy_top + row * cell
            # reportlab y: bottom-left origin
            y0_rl = ph - y0_top - cell
            # cell border
            c.setStrokeColorRGB(0, 0, 0)
            c.setFillColorRGB(1, 1, 1)
            c.rect(x0, y0_rl, cell, cell, stroke=1, fill=1)
            # centered text
            text = entry_labels[int(labels[row, col])]
            tw = pdfmetrics.stringWidth(text, font_name, font_size)
            tx = x0 + (cell - tw) / 2.0
            ty = y0_rl + (cell - text_h) / 2.0 - descent_pt
            c.setFillColorRGB(0, 0, 0)
            c.drawString(tx, ty, text)

    # --- Legend ---
    swatch = lay["swatch"]
    legend_font_name = "Helvetica"
    legend_font_size = max(8.0, swatch / 2.0)
    c.setFont(legend_font_name, legend_font_size)

    for slot, i in enumerate(order):
        color = palette[i]
        col = slot % lay["legend_cols"]
        row = slot // lay["legend_cols"]
        x0 = lay["legend_x"] + col * lay["legend_col_w"]
        y0_top = lay["legend_y"] + row * (swatch + lay["legend_gap"])
        y0_rl = ph - y0_top - swatch

        # swatch
        c.setFillColorRGB(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(border)
        c.rect(x0, y0_rl, swatch, swatch, stroke=1, fill=1)

        # label
        c.setFillColorRGB(0, 0, 0)
        c.setFont(legend_font_name, legend_font_size)
        lx = x0 + swatch + swatch / 4.0
        ly = y0_rl + swatch / 4.0
        c.drawString(lx, ly, entry_labels[i])

    c.save()
    return buf.getvalue()


def render_solution_pdf(
    labels: np.ndarray,
    palette: np.ndarray,
    lay: dict,
    page,
) -> bytes:
    """Render a filled-in solution preview as a vector PDF."""
    h, w = labels.shape
    pw, ph = page.size_pt
    cell = lay["cell_pt"]
    border = lay["line_width"] if lay.get("line_width") is not None else max(0.5, cell * 0.04)
    gx, gy_top = lay["grid_x"], lay["grid_y"]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(pw, ph))
    c.setLineWidth(border)

    for row in range(h):
        for col in range(w):
            x0 = gx + col * cell
            y0_top = gy_top + row * cell
            y0_rl = ph - y0_top - cell
            color = palette[int(labels[row, col])]
            c.setFillColorRGB(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
            c.setStrokeColorRGB(0, 0, 0)
            c.rect(x0, y0_rl, cell, cell, stroke=1, fill=1)

    c.save()
    return buf.getvalue()
