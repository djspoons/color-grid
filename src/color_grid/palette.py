"""Load color-set files in the project's simple JSON format.

Schema:
    {
        "name": "<display name>",
        "color_space": "srgb" | "lab" | "ciecam16_ucs",
        "colors": [
            {"code": "<short label>", "color": [<3 numbers>]},
            ...
        ]
    }

- `srgb` entries are [r, g, b] uint values 0..255.
- `lab` entries are CIELAB [L, a, b] floats (L in 0..100).
- `ciecam16_ucs` entries are CAM16-UCS [J, a, b] floats.

The loader converts to sRGB uint8 at load time for rendering. Clustering code
still converts sRGB to its active working space on demand.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.color import lab2rgb


_VALID_SPACES = ("srgb", "lab", "ciecam16_ucs")


@dataclass(frozen=True)
class Palette:
    name: str
    rgb: np.ndarray  # (P, 3) uint8
    codes: list[str]


def load_palette(path: Path) -> Palette:
    path = Path(path)
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")

    name = str(doc.get("name", path.stem))
    space = doc.get("color_space")
    if space not in _VALID_SPACES:
        raise ValueError(
            f"{path}: color_space must be one of {_VALID_SPACES}, got {space!r}"
        )

    entries = doc.get("colors")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: 'colors' must be a non-empty array")

    codes: list[str] = []
    values: list[list[float]] = []
    for i, entry in enumerate(entries):
        try:
            codes.append(str(entry["code"]))
            color = entry["color"]
            if len(color) != 3:
                raise ValueError("expected 3 components")
            values.append([float(x) for x in color])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: entry {i} is malformed: {e}") from e

    arr = np.asarray(values, dtype=np.float64)
    rgb = _to_srgb_u8(arr, space)
    return Palette(name=name, rgb=rgb, codes=codes)


def _to_srgb_u8(values: np.ndarray, color_space: str) -> np.ndarray:
    """Convert a (P, 3) color array in the given space to uint8 sRGB."""
    if color_space == "srgb":
        return np.clip(values, 0, 255).astype(np.uint8)
    if color_space == "lab":
        rgb01 = lab2rgb(values.reshape(1, -1, 3)).reshape(-1, 3)
        return np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    # ciecam16_ucs. The file uses the standard 0..100 J/a/b scale, but
    # colour.convert expects a normalized 0..1 scale, so divide first.
    import colour

    jab01 = values / 100.0
    rgb01 = colour.convert(
        jab01.reshape(1, -1, 3), "CAM16UCS", "Output-Referred RGB"
    )
    rgb01 = np.asarray(rgb01, dtype=np.float64).reshape(-1, 3)
    return np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
