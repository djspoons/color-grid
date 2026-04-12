"""Assign single-character keys to palette colors for grid labeling."""

import colorsys

import numpy as np

# Canonical hues in HSV (0..1) for primary rainbow letters.
_RAINBOW = [
    ("R", 0.000),  # red
    ("O", 0.083),  # orange
    ("Y", 0.167),  # yellow
    ("G", 0.333),  # green
    ("B", 0.667),  # blue
    ("I", 0.764),  # indigo
    ("V", 0.833),  # violet
]

# Secondary chromatic letters for hues not well covered by ROYGBIV.
_SECONDARY = [
    ("L", 0.250),  # lime (yellow-green)
    ("T", 0.500),  # teal/cyan
    ("P", 0.917),  # pink/magenta
]

# Maximum hue distance (0..1) to qualify for a letter.
_HUE_THRESHOLD = 0.07

# Saturation below which a color is considered neutral.
_NEUTRAL_SAT = 0.20

# Value thresholds for K (black) and W (white).
_BLACK_V = 0.15
_NEAR_BLACK_V = 0.30
_NEAR_BLACK_S = 0.35
_WHITE_V = 0.92
_WHITE_S = 0.05


def _hue_dist(a: float, b: float) -> float:
    """Circular distance between two hues in 0..1."""
    d = abs(a - b)
    return min(d, 1.0 - d)


def assign_keys(palette: np.ndarray) -> list[str]:
    """Assign a unique single-character key to each palette entry.

    Args:
        palette: (n, 3) uint8 sRGB array.

    Returns:
        List of n single-character strings.
    """
    n = len(palette)
    keys: list[str | None] = [None] * n

    # Convert to HSV.
    hsv = []
    for rgb in palette:
        r, g, b = float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0
        hsv.append(colorsys.rgb_to_hsv(r, g, b))

    chromatic = []  # (index, hue, sat, val)
    neutral = []    # (index, sat, val)
    for i, (h, s, v) in enumerate(hsv):
        # Treat as neutral if low saturation, very dark, or dark + low saturation.
        if s < _NEUTRAL_SAT or v < _BLACK_V or (v < _NEAR_BLACK_V and s < _NEAR_BLACK_S):
            neutral.append((i, s, v))
        else:
            chromatic.append((i, h, s, v))

    # --- Assign K and W first ---
    used_letters: set[str] = set()

    # K: darkest neutral, only if very dark.
    if neutral:
        darkest = min(neutral, key=lambda x: x[2])
        if darkest[2] < _BLACK_V:
            keys[darkest[0]] = "K"
            used_letters.add("K")
            neutral = [x for x in neutral if x[0] != darkest[0]]

    # W: lightest neutral, only if very light and desaturated.
    if neutral:
        lightest = max(neutral, key=lambda x: x[2])
        _, ls, lv = lightest
        if lv > _WHITE_V and ls < _WHITE_S:
            keys[lightest[0]] = "W"
            used_letters.add("W")
            neutral = [x for x in neutral if x[0] != lightest[0]]

    # --- Assign chromatic letters (ROYGBIV, then secondary) ---
    candidates = _RAINBOW + _SECONDARY
    # Build (letter, canonical_hue, color_index, distance) pairs.
    matches = []
    for letter, canon_hue in candidates:
        for i, h, s, v in chromatic:
            if keys[i] is not None:
                continue
            d = _hue_dist(h, canon_hue)
            if d < _HUE_THRESHOLD:
                matches.append((d, letter, i))

    # Greedy assignment: best matches first.
    matches.sort()
    assigned_indices: set[int] = set()
    for d, letter, i in matches:
        if letter in used_letters or i in assigned_indices:
            continue
        keys[i] = letter
        used_letters.add(letter)
        assigned_indices.add(i)

    # --- Assign neutral digits ---
    # Determine which digits to skip (avoid 1 if I used, 0 if O used).
    skip_digits: set[str] = set()
    if "I" in used_letters:
        skip_digits.add("1")
    if "O" in used_letters:
        skip_digits.add("0")

    available_digits = [d for d in "0123456789" if d not in skip_digits]
    # Sort neutrals by value (darkest first = lowest digit).
    neutral.sort(key=lambda x: x[2])
    for idx, (i, s, v) in enumerate(neutral):
        if idx < len(available_digits):
            keys[i] = available_digits[idx]
            used_letters.add(available_digits[idx])
        # If more than 10 neutrals, fall through to the fallback below.

    # --- Fallback for any remaining unassigned colors ---
    # Prefer letters that aren't easily confused with the heuristic keys.
    _HEURISTIC_LETTERS = set("ROYGBIVKWPTLM")
    fallback_preferred = [c for c in "ACDEFJNQSUXZ" if c not in _HEURISTIC_LETTERS]
    fallback_heuristic = [c for c in "ROYGBIVKWPTLM"]
    fallback_all = fallback_preferred + fallback_heuristic
    fallback_lower = [c.lower() for c in fallback_all]
    fallback_pool = [c for c in fallback_all + fallback_lower if c not in used_letters]
    fi = 0
    for i in range(n):
        if keys[i] is None:
            if fi < len(fallback_pool):
                keys[i] = fallback_pool[fi]
                used_letters.add(fallback_pool[fi])
                fi += 1
            else:
                keys[i] = "?"

    return keys  # type: ignore[return-value]
