import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans


ColorSpace = str  # "rgb", "lab", or "ciecam16"
Method = str  # "kmeans" or "maxcoverage"

_VALID_SPACES = ("rgb", "lab", "ciecam16")


def quantize_cells(
    cells: np.ndarray,
    n_colors: int,
    color_space: ColorSpace = "lab",
    method: Method = "kmeans",
    fixed_palette: np.ndarray | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster cell colors into `n_colors` groups.

    Args:
        cells: (H, W, 3) float RGB array in 0..255.
        n_colors: number of palette colors.
        color_space: "rgb", "lab", or "ciecam16" (CIECAM16-UCS — slowest but
            closest to perceptual difference).
        method: "kmeans" minimizes average error; "maxcoverage" uses farthest-
            first selection to preserve rare vivid colors.
        fixed_palette: optional (P, 3) uint8 sRGB array. When provided, each
            cluster center is snapped to its nearest unused palette entry.
    """
    if color_space not in _VALID_SPACES:
        raise ValueError(f"unknown color_space: {color_space!r}")
    if method not in ("kmeans", "maxcoverage"):
        raise ValueError(f"unknown method: {method!r}")

    h, w, _ = cells.shape
    features = _rgb_to_features(cells, color_space)

    n_unique = len(np.unique(features, axis=0))
    k = min(n_colors, n_unique)
    if k < 1:
        raise ValueError("need at least one cell to quantize")

    if method == "kmeans":
        labels, centers = _kmeans(features, k, random_state)
    else:
        labels, centers = _farthest_first(features, k)

    if fixed_palette is not None:
        labels, palette = _snap_to_palette(
            features, centers, fixed_palette, color_space
        )
        return labels.reshape(h, w), palette

    rgb_centers = _features_to_rgb(centers, color_space)
    palette = np.clip(rgb_centers, 0, 255).astype(np.uint8)
    return labels.reshape(h, w), palette


_colour_mod = None


def _colour():
    global _colour_mod
    if _colour_mod is None:
        import colour  # noqa: F401

        _colour_mod = colour
    return _colour_mod


def _rgb_to_features(rgb_255: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    """Convert an RGB array (any shape, last dim 3, values 0..255) to a flat
    (N, 3) feature array in the given color space."""
    if color_space == "rgb":
        return rgb_255.reshape(-1, 3).astype(np.float32)

    rgb01 = np.clip(rgb_255.astype(np.float32) / 255.0, 0.0, 1.0)
    if color_space == "lab":
        return rgb2lab(rgb01).reshape(-1, 3)
    # ciecam16: expects shape (..., 3); preserve shape, then flatten.
    jab = _colour().convert(rgb01, "Output-Referred RGB", "CAM16UCS")
    return np.asarray(jab, dtype=np.float32).reshape(-1, 3)


def _features_to_rgb(features: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    """Convert a (N, 3) feature array back to a (N, 3) float RGB array in 0..255."""
    if color_space == "rgb":
        return features
    if color_space == "lab":
        return lab2rgb(features.reshape(1, -1, 3)).reshape(-1, 3) * 255.0
    rgb01 = _colour().convert(
        features.reshape(1, -1, 3), "CAM16UCS", "Output-Referred RGB"
    )
    return np.asarray(rgb01, dtype=np.float32).reshape(-1, 3) * 255.0


def _kmeans(features: np.ndarray, k: int, random_state: int):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(features)
    return labels, km.cluster_centers_


def _farthest_first(features: np.ndarray, k: int):
    """Farthest-first traversal (a.k.a. k-centers / max-min).

    Deterministically seeds from the point farthest from the centroid, then
    repeatedly picks the point whose minimum distance to the current set of
    centers is maximum. Each picked point is a real cell color, so extremes
    are always preserved.
    """
    centroid = features.mean(axis=0)
    dists = np.linalg.norm(features - centroid, axis=1)
    first = int(np.argmax(dists))

    center_idx = [first]
    min_dist = np.linalg.norm(features - features[first], axis=1)

    for _ in range(k - 1):
        i = int(np.argmax(min_dist))
        center_idx.append(i)
        new_d = np.linalg.norm(features - features[i], axis=1)
        min_dist = np.minimum(min_dist, new_d)

    centers = features[center_idx]
    diffs = features[:, None, :] - centers[None, :, :]
    labels = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)
    return labels, centers


def _snap_to_palette(
    features: np.ndarray,
    centers: np.ndarray,
    fixed_palette: np.ndarray,
    color_space: ColorSpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Map cluster centers to their nearest unused palette entries.

    features and centers are in `color_space`; fixed_palette is uint8 sRGB.
    """
    palette_features = _rgb_to_features(fixed_palette, color_space)

    used: set[int] = set()
    chosen: list[int] = []
    for center in centers:
        dists = np.linalg.norm(palette_features - center, axis=1)
        order = np.argsort(dists)
        for idx in order:
            idx = int(idx)
            if idx not in used:
                used.add(idx)
                chosen.append(idx)
                break
        else:
            raise ValueError("palette smaller than requested number of colors")

    chosen_features = palette_features[chosen]
    diffs = features[:, None, :] - chosen_features[None, :, :]
    labels = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)
    snapped_rgb = fixed_palette[chosen].astype(np.uint8)
    return labels, snapped_rgb
