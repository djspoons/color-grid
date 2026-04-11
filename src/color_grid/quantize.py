import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans


ColorSpace = str  # "rgb" or "lab"


def quantize_cells(
    cells: np.ndarray,
    n_colors: int,
    color_space: ColorSpace = "rgb",
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster cell colors into `n_colors` groups.

    Args:
        cells: (H, W, 3) float RGB array in 0..255.
        n_colors: number of palette colors.
        color_space: "rgb" clusters in sRGB; "lab" clusters in CIE LAB,
            which is closer to perceptual color difference.

    Returns:
        labels: (H, W) int array of palette indices.
        palette: (n_colors, 3) uint8 RGB array, ordered by cluster index.
    """
    if color_space not in ("rgb", "lab"):
        raise ValueError(f"unknown color_space: {color_space!r}")

    h, w, _ = cells.shape

    if color_space == "lab":
        rgb01 = np.clip(cells / 255.0, 0.0, 1.0)
        features = rgb2lab(rgb01).reshape(-1, 3)
    else:
        features = cells.reshape(-1, 3)

    n_unique = len(np.unique(features, axis=0))
    k = min(n_colors, n_unique)
    if k < 1:
        raise ValueError("need at least one cell to quantize")

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_

    if color_space == "lab":
        rgb_centers = lab2rgb(centers.reshape(1, -1, 3)).reshape(-1, 3) * 255.0
    else:
        rgb_centers = centers

    palette = np.clip(rgb_centers, 0, 255).astype(np.uint8)
    return labels.reshape(h, w), palette
