import json
from pathlib import Path

import numpy as np


def load_palette(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load a fixed color palette from a JSON file.

    Currently supports the color-set format used under color-sets/, where the
    file is a list of entries with `color.srgb.{r,g,b}` and a display name in
    `color.color` (e.g. ["Blue", "B2"]).

    Returns:
        rgb: (P, 3) uint8 array.
        names: length-P list of display names.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of color entries")

    rgb: list[list[int]] = []
    names: list[str] = []
    for i, entry in enumerate(data):
        try:
            srgb = entry["color"]["srgb"]
            rgb.append([int(srgb["r"]), int(srgb["g"]), int(srgb["b"])])
        except (KeyError, TypeError) as e:
            raise ValueError(f"{path}: entry {i} missing color.srgb.{{r,g,b}}") from e

        name_field = entry.get("color", {}).get("color")
        if isinstance(name_field, list):
            name = " ".join(str(x) for x in name_field)
        elif isinstance(name_field, str):
            name = name_field
        else:
            name = f"#{i+1}"
        names.append(name)

    return np.array(rgb, dtype=np.uint8), names
