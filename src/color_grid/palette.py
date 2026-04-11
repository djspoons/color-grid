import json
import re
import sys
from pathlib import Path

import numpy as np


def load_palette(
    path: Path,
) -> tuple[np.ndarray, list[str], list[str | None]]:
    """Load a fixed color palette from a JSON file.

    Supports the color-set format used under color-sets/, where the file is a
    list of entries with `color.srgb.{r,g,b}` and a family descriptor in
    `color.color` (e.g. ["Blue", "B2"] or ["A", "Blue", "B3"] for accents).

    If a sibling HTML file (same base name, ``.html``) exists, each entry is
    additionally matched by its CIELAB value to the HTML row's ``data-code``
    attribute — giving the real pencil number (e.g. ``"701"``) as the label.

    Returns:
        rgb: (P, 3) uint8 array.
        families: length-P list of full family names ("Blue", "Turquoise", …),
            or "" when one can't be determined.
        codes: length-P list; each element is the pencil code from the HTML
            sidecar, or None when no sidecar was found or the entry could not
            be matched. Use `make_subset_labels(families[selected])` as a
            fallback when codes are unavailable.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of color entries")

    rgb: list[list[int]] = []
    families: list[str] = []
    labs: list[tuple[float, float, float] | None] = []
    for i, entry in enumerate(data):
        color = entry.get("color", {})
        try:
            srgb = color["srgb"]
            rgb.append([int(srgb["r"]), int(srgb["g"]), int(srgb["b"])])
        except (KeyError, TypeError) as e:
            raise ValueError(f"{path}: entry {i} missing color.srgb.{{r,g,b}}") from e
        families.append(_extract_family(color.get("color")))

        cielab = color.get("cielab") or {}
        try:
            labs.append(
                (
                    float(cielab["l"]),
                    float(cielab["a"]),
                    float(cielab["b"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            labs.append(None)

    codes = _codes_from_sidecar(path, labs)
    return np.array(rgb, dtype=np.uint8), families, codes


def _codes_from_sidecar(
    json_path: Path, labs: list[tuple[float, float, float] | None]
) -> list[str | None]:
    """Look for a `<basename>.html` alongside the JSON file and match each
    palette entry to an HTML row by CIELAB value. Returns per-entry codes or
    a list of Nones if no sidecar is present."""
    html_path = json_path.with_suffix(".html")
    if not html_path.exists():
        return [None] * len(labs)
    try:
        html_entries = _parse_html_codes(html_path)
    except Exception as e:
        print(
            f"warning: could not parse {html_path.name}: {e}",
            file=sys.stderr,
        )
        return [None] * len(labs)

    # First pass: exact LAB tuple lookup (rounded to 3 decimals, which is the
    # precision the color-sets files use).
    by_lab: dict[tuple[float, float, float], str] = {}
    for code, lab in html_entries:
        by_lab[_round_lab(lab)] = code

    codes: list[str | None] = []
    missing: list[int] = []
    for i, lab in enumerate(labs):
        if lab is None:
            codes.append(None)
            continue
        hit = by_lab.get(_round_lab(lab))
        codes.append(hit)
        if hit is None:
            missing.append(i)

    # Second pass: nearest-neighbor for any unmatched entries. Uses Hungarian
    # assignment so multiple near-duplicates don't all snap to the same code.
    if missing:
        from scipy.optimize import linear_sum_assignment

        miss_labs = np.array([labs[i] for i in missing])
        html_labs = np.array([lab for _, lab in html_entries])
        html_codes = [c for c, _ in html_entries]
        cost = np.linalg.norm(miss_labs[:, None, :] - html_labs[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            codes[missing[r]] = html_codes[c]

    return codes


def _round_lab(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    return (round(lab[0], 3), round(lab[1], 3), round(lab[2], 3))


_MASSTONE_RE = re.compile(
    r'id="masstone-([a-z0-9]+)"[^>]*>\s*<rect[^>]*fill="lab\(([-\d. ]+)\)"',
    re.DOTALL,
)
_PAINT_CODE_RE = re.compile(
    r'data-paint-id="([a-z0-9]+)"[^>]*?data-code="(\d+)"',
    re.DOTALL,
)


def _parse_html_codes(html_path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    """Extract (data-code, cielab) pairs from the color-set HTML sidecar.

    Each "masstone" svg carries the canonical LAB color for one pencil, and
    each enclosing div carries its data-code. The same paint id may appear in
    multiple views (masstone/undertone/etc.); we take the masstone value.
    """
    html = html_path.read_text()
    paint_to_code: dict[str, str] = {}
    for paint_id, code in _PAINT_CODE_RE.findall(html):
        paint_to_code.setdefault(paint_id, code)

    entries: list[tuple[str, tuple[float, float, float]]] = []
    for paint_id, lab_str in _MASSTONE_RE.findall(html):
        code = paint_to_code.get(paint_id)
        if code is None:
            continue
        parts = lab_str.split()
        if len(parts) != 3:
            continue
        lab = (float(parts[0]), float(parts[1]), float(parts[2]))
        entries.append((code, lab))
    return entries


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
