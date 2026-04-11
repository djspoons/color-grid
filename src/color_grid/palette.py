import json
import sys
from pathlib import Path

import numpy as np


def load_palette(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load a fixed color palette from a JSON file.

    Supports the color-set format used under color-sets/, where the file is a
    list of entries with `color.srgb.{r,g,b}` and a family descriptor in
    `color.color` (e.g. ["Blue", "B2"] or ["A", "Blue", "B3"] for accents).

    Returns:
        rgb: (P, 3) uint8 array.
        families: length-P list of full family names ("Blue", "Turquoise", …),
            or "" when one can't be determined. Use `make_subset_labels` to
            turn a selected subset of these into short grid labels.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of color entries")

    rgb: list[list[int]] = []
    families: list[str] = []
    for i, entry in enumerate(data):
        color = entry.get("color", {})
        try:
            srgb = color["srgb"]
            rgb.append([int(srgb["r"]), int(srgb["g"]), int(srgb["b"])])
        except (KeyError, TypeError) as e:
            raise ValueError(f"{path}: entry {i} missing color.srgb.{{r,g,b}}") from e
        families.append(_extract_family(color.get("color")))

    return np.array(rgb, dtype=np.uint8), families


def _extract_family(name_field) -> str:
    """Derive the family name from a color.color field.

    ["Blue", "B2"] -> "Blue"
    ["A", "Blue", "B3"] -> "Blue"   (skip the single-letter accent prefix)
    """
    if not isinstance(name_field, list) or not name_field:
        return ""
    items = [str(x) for x in name_field]
    if len(items) >= 3 and len(items[0]) == 1:
        return items[1]
    return items[0]


def make_subset_labels(families: list[str]) -> list[str]:
    """Build short unique labels for a selected subset of palette entries.

    Format: first letter of the family + 1-based index within that family.
    e.g. ["Blue", "Blue", "Red", "Green"] -> ["B1", "B2", "R1", "G1"].

    If two selected families start with the same letter, fall back to plain
    sequential numbers ("1", "2", ...) for the whole subset and warn, since
    single-letter labels would be ambiguous.
    """
    families = [f or "" for f in families]
    letters = {f[0].upper() for f in families if f}
    letter_to_family: dict[str, str] = {}
    collision = False
    for f in families:
        if not f:
            continue
        letter = f[0].upper()
        if letter in letter_to_family and letter_to_family[letter] != f:
            collision = True
            break
        letter_to_family[letter] = f

    if collision or any(not f for f in families):
        if collision:
            print(
                "warning: palette families collide on first letter "
                f"({sorted(letters)}); falling back to sequential labels",
                file=sys.stderr,
            )
        return [str(i + 1) for i in range(len(families))]

    counts: dict[str, int] = {}
    labels: list[str] = []
    for f in families:
        letter = f[0].upper()
        counts[letter] = counts.get(letter, 0) + 1
        labels.append(f"{letter}{counts[letter]}")
    return labels
