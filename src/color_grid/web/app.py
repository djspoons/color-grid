"""FastAPI application for the color-grid web UI."""

import io
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from ..grid import image_to_cell_colors
from ..palette import load_palette
from ..quantize import quantize_cells
from ..render import PAPER_SIZES_INCHES, PageSpec, render_page, render_solution
from .state import Session, create_session, get_session

_WEB_DIR = Path(__file__).resolve().parent
_PALETTES_DIR = _WEB_DIR.parent.parent.parent / "palettes"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="Color Grid")
app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=_WEB_DIR / "templates")


def _discover_palettes() -> list[dict]:
    """Scan palettes directory for JSON files."""
    if not _PALETTES_DIR.is_dir():
        return []
    palettes = []
    for p in sorted(_PALETTES_DIR.glob("*.json")):
        palettes.append({"filename": p.name, "label": p.stem.replace("-", " ").title()})
    return palettes


_PALETTES = _discover_palettes()


def _session_from_request(request: Request) -> Session | None:
    sid = request.cookies.get("colorgrid_session")
    return get_session(sid)


def _set_session_cookie(response: Response, session: Session) -> None:
    response.set_cookie("colorgrid_session", session.id, httponly=True, samesite="lax")


def _clamp(val: int | float, lo: int | float, hi: int | float):
    return max(lo, min(hi, val))


def _error(request: Request, message: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/error.html", {"message": message}
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session = _session_from_request(request)
    has_image = session is not None and session.image is not None
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "has_image": has_image,
            "palettes": _PALETTES,
            "paper_sizes": sorted(PAPER_SIZES_INCHES.keys()),
        },
    )
    if session is None:
        session = create_session()
        _set_session_cookie(response, session)
    return response


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, image: UploadFile):
    session = _session_from_request(request)
    if session is None:
        session = create_session()

    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        resp = _error(request, "Image too large (10 MB max).")
        _set_session_cookie(resp, session)
        return resp

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Exception):
        resp = _error(request, "Could not read image. Please upload a JPG or PNG.")
        _set_session_cookie(resp, session)
        return resp

    img = img.convert("RGB")
    session.image = img
    if image.filename:
        session.image_stem = Path(image.filename).stem
    session.labels = None
    session.palette = None

    w, h = img.size
    resp = templates.TemplateResponse(
        request, "partials/upload_ok.html", {"width": w, "height": h},
    )
    _set_session_cookie(resp, session)
    return resp


@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    width: int = Form(20),
    height: int = Form(20),
    colors: int = Form(8),
    color_mode: str = Form("count"),
    color_space: str = Form("lab"),
    method: str = Form("maxcoverage"),
    palette_name: str = Form(""),
    paper: str = Form("letter"),
    margin: float = Form(0.5),
):
    session = _session_from_request(request)
    if session is None or session.image is None:
        return _error(request, "Please upload an image first.")

    width = int(_clamp(width, 12, 72))
    height = int(_clamp(height, 12, 72))
    colors = int(_clamp(colors, 2, 30))
    margin = float(_clamp(margin, 0.1, 2.0))

    if color_space not in ("rgb", "lab", "ciecam16"):
        color_space = "lab"
    if method not in ("kmeans", "maxcoverage"):
        method = "maxcoverage"
    if paper not in PAPER_SIZES_INCHES:
        paper = "letter"

    fixed_palette = None
    palette_codes = None
    if color_mode == "palette" and palette_name:
        pal_path = _PALETTES_DIR / palette_name
        if not pal_path.is_file():
            return _error(request, f"Palette file not found: {palette_name}")
        pal = load_palette(pal_path)
        fixed_palette = pal.rgb
        palette_codes = pal.codes
        colors = len(fixed_palette)

    try:
        cells = await run_in_threadpool(
            image_to_cell_colors, session.image, width, height
        )
        labels, palette, chosen_indices = await run_in_threadpool(
            quantize_cells,
            cells,
            colors,
            color_space=color_space,
            method=method,
            fixed_palette=fixed_palette,
        )
    except ValueError as e:
        return _error(request, str(e))

    entry_labels = None
    if chosen_indices is not None and palette_codes is not None:
        entry_labels = [palette_codes[int(i)] for i in chosen_indices]

    page_spec = PageSpec(paper=paper, margin_in=margin)

    try:
        grid_svg = await run_in_threadpool(
            render_page, labels, palette, page_spec, entry_labels, "svg"
        )
        solution_svg = await run_in_threadpool(
            render_solution, labels, palette, page_spec, "svg"
        )
    except ValueError as e:
        return _error(request, str(e))

    session.labels = labels
    session.palette = palette
    session.entry_labels = entry_labels
    session.page_spec = page_spec

    return templates.TemplateResponse(
        request,
        "partials/preview.html",
        {
            "grid_svg": grid_svg.decode("utf-8"),
            "solution_svg": solution_svg.decode("utf-8"),
        },
    )


@app.get("/download/grid.pdf")
async def download_grid_pdf(request: Request):
    session = _session_from_request(request)
    if session is None or session.labels is None:
        return Response("No grid generated yet.", status_code=404)

    pdf_data = await run_in_threadpool(
        render_page,
        session.labels,
        session.palette,
        session.page_spec,
        session.entry_labels,
        "pdf",
    )
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={session.image_stem}_grid.pdf"},
    )


@app.get("/download/solution.pdf")
async def download_solution_pdf(request: Request):
    session = _session_from_request(request)
    if session is None or session.labels is None:
        return Response("No grid generated yet.", status_code=404)

    pdf_data = await run_in_threadpool(
        render_solution,
        session.labels,
        session.palette,
        session.page_spec,
        "pdf",
    )
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={session.image_stem}_solution.pdf"},
    )
