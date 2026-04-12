import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from color_grid.web.app import app

client = TestClient(app)


def _make_test_image() -> bytes:
    """Create a small test JPEG in memory."""
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :20] = (255, 0, 0)
    arr[:20, 20:] = (0, 255, 0)
    arr[20:, :20] = (0, 0, 255)
    arr[20:, 20:] = (255, 255, 0)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_index_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Color Grid" in resp.text
    assert "colorgrid_session" in resp.cookies


def test_upload_and_generate():
    img_bytes = _make_test_image()

    # Upload
    resp = client.post(
        "/upload",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert "Image loaded" in resp.text

    # Generate
    resp = client.post(
        "/generate",
        data={
            "width": "4",
            "height": "4",
            "colors": "4",
            "color_space": "rgb",
            "method": "maxcoverage",
            "palette_name": "none",
            "paper": "letter",
            "margin": "0.5",
        },
    )
    assert resp.status_code == 200
    assert "<svg" in resp.text
    assert "Download Grid PDF" in resp.text

    # Parse both SVGs from the response
    # The response contains two SVGs embedded in HTML
    assert resp.text.count("<svg") == 2


def test_pdf_download_after_generate():
    img_bytes = _make_test_image()

    client.post("/upload", files={"image": ("test.jpg", img_bytes, "image/jpeg")})
    client.post(
        "/generate",
        data={
            "width": "4",
            "height": "4",
            "colors": "3",
            "color_space": "rgb",
            "method": "kmeans",
            "palette_name": "none",
            "paper": "letter",
            "margin": "0.5",
        },
    )

    # Grid PDF
    resp = client.get("/download/grid.pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:5] == b"%PDF-"

    # Solution PDF
    resp = client.get("/download/solution.pdf")
    assert resp.status_code == 200
    assert resp.content[:5] == b"%PDF-"


def test_download_without_generate():
    # Fresh client with no session data
    fresh = TestClient(app, cookies={})
    resp = fresh.get("/download/grid.pdf")
    assert resp.status_code == 404


def test_upload_invalid_file():
    resp = client.post(
        "/upload",
        files={"image": ("bad.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 200  # returns error fragment, not HTTP error
    assert "Could not read image" in resp.text


def test_generate_without_upload():
    fresh = TestClient(app, cookies={})
    resp = fresh.post(
        "/generate",
        data={"width": "4", "height": "4", "colors": "4"},
    )
    assert resp.status_code == 200
    assert "upload an image" in resp.text.lower()
