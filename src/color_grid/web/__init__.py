"""Color Grid web interface."""

import os

from .app import app


def run() -> None:
    """Start the server."""
    import uvicorn

    host = os.environ.get("COLORGRID_HOST", "127.0.0.1")
    port = int(os.environ.get("COLORGRID_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
