import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans


ColorSpace = str  # "rgb" or "lab"
Method = str  # "kmeans" or "maxcoverage"


def quantize_cells(
    cells: np.ndarray,
    n_colors: int,
    color_space: ColorSpace = "lab",
    method: Method = "kmeans",
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster cell colors into `n_colors` groups.

    Args:
        cells: (H, W, 3) float RGB array in 0..255.
        n_colors: number of palette colors.
        color_space: "rgb" or "lab" (CIE LAB is closer to perceptual difference).
        method: "kmeans" minimizes average error and tends to lose rare colors;
            "maxcoverage" uses farthest-first selection to preserve the full
            color range, so small vivid regions (e.g. a red eye) survive.
    """
    if color_space not in ("rgb", "lab"):
        raise ValueError(f"unknown color_space: {color_space!r}")
    if method not in ("kmeans", "maxcoverage"):
        raise ValueError(f"unknown method: {method!r}")

    h, w, _ = cells.shape

    if color_space == "lab":
        rgb01 = np.clip(cells / 255.0, 0.0, 1.0)
        features = rgb2lab(rgb01).reshape(-1, 3)
    else:
        features = cells.reshape(-1, 3).astype(np.float32)

    n_unique = len(np.unique(features, axis=0))
    k = min(n_colors, n_unique)
    if k < 1:
        raise ValueError("need at least one cell to quantize")

    if method == "kmeans":
        labels, centers = _kmeans(features, k, random_state)
    else:
        labels, centers = _farthest_first(features, k)

    if color_space == "lab":
        rgb_centers = lab2rgb(centers.reshape(1, -1, 3)).reshape(-1, 3) * 255.0
    else:
        rgb_centers = centers

    palette = np.clip(rgb_centers, 0, 255).astype(np.uint8)
    return labels.reshape(h, w), palette


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
    n = len(features)
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
    # Assign every feature to its nearest center.
    diffs = features[:, None, :] - centers[None, :, :]
    labels = np.argmin(np.linalg.norm(diffs, axis=2), axis=1)
    return labels, centers
