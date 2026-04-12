"""In-memory session store for the web UI."""

import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..render import PageSpec

SESSION_TTL = 30 * 60  # 30 minutes

_store: dict[str, "Session"] = {}


@dataclass
class Session:
    id: str
    image: Image.Image | None = None
    labels: np.ndarray | None = None
    palette: np.ndarray | None = None
    entry_labels: list[str] | None = None
    page_spec: PageSpec | None = None
    created_at: float = field(default_factory=time.time)


def create_session() -> Session:
    """Create a new session and add it to the store."""
    _prune_expired()
    sid = uuid.uuid4().hex
    session = Session(id=sid)
    _store[sid] = session
    return session


def get_session(sid: str | None) -> Session | None:
    """Look up a session by ID, returning None if missing or expired."""
    if sid is None:
        return None
    session = _store.get(sid)
    if session is None:
        return None
    if time.time() - session.created_at > SESSION_TTL:
        _store.pop(sid, None)
        return None
    return session


def _prune_expired() -> None:
    """Remove expired sessions."""
    now = time.time()
    expired = [k for k, v in _store.items() if now - v.created_at > SESSION_TTL]
    for k in expired:
        del _store[k]
