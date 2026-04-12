"""Color Grid web interface."""

from .app import app


def run() -> None:
    """Start the development server."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
