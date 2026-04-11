import numpy as np
from sklearn.cluster import KMeans


def quantize_cells(
    cells: np.ndarray, n_colors: int, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster cell colors into `n_colors` groups.

    Args:
        cells: (H, W, 3) float RGB array.
        n_colors: number of palette colors.

    Returns:
        labels: (H, W) int array of palette indices.
        palette: (n_colors, 3) uint8 RGB array, ordered by cluster index.
    """
    h, w, _ = cells.shape
    flat = cells.reshape(-1, 3)

    n_unique = len(np.unique(flat, axis=0))
    k = min(n_colors, n_unique)
    if k < 1:
        raise ValueError("need at least one cell to quantize")

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(flat)
    centers = kmeans.cluster_centers_

    palette = np.clip(centers, 0, 255).astype(np.uint8)
    return labels.reshape(h, w), palette
