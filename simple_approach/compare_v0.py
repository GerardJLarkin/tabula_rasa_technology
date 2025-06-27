import numpy as np
from typing import Any, List, Union

def compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
    """
    Compute a weighted similarity score between two 'patom' arrays.
    Assumes columns:
      0: patom_id (any hashable)
      3–4: normalized x,y positions
      5: colour
    Returns [id_a, id_b, similarity_score].
    """
    m, n = a.shape[0], b.shape[0]
    # Early exit on empties
    if m == 0 or n == 0:
        return [None, None, np.nan]

    # Extract and cast once to float32 (no full copy if already float32)
    pos1 = a[:, 3:5].astype(np.float32, copy=False)
    pos2 = b[:, 3:5].astype(np.float32, copy=False)
    col1 = a[:, 5].astype(np.float32, copy=False)
    col2 = b[:, 5].astype(np.float32, copy=False)

    # Sum of norms squared
    sum_sq1 = np.einsum('ij,ij->', pos1, pos1)      # ∑‖x_i‖²
    sum_sq2 = np.einsum('ij,ij->', pos2, pos2)      # ∑‖y_j‖²
    # Sum of all dot-products
    dot_sum = np.sum(pos1.dot(pos2.T))             # ∑_i ∑_j x_i·y_j

    # Total pairwise squared-distance sum
    total_sq = m * sum_sq2 + n * sum_sq1 - 2 * dot_sum
    pos_sim = np.sqrt(max(total_sq, 0.0)) / (m * n)

    # Colour similarity: sum |c_i - d_j| over all i,j
    # This still allocates an (m×n) diff matrix, but colour vectors are 1-D.
    # If m×n is huge you could instead sort & use convolution-like tricks,
    # but often colour arrays are small enough to handle.
    colour_sim = np.abs(col1[:, None] - col2[None, :]).sum() / (m * n)

    # Pixel-count similarity
    fill_diff = abs(m - n) / ((m + n) / 2)
    fill_sim = min(fill_diff, 1.0)

    # IDs
    id_a, id_b = a[0, 0], b[0, 0]

    # Weighted combination
    total = pos_sim + colour_sim + fill_sim
    if total == 0.0:
        score = 1.0 / 3.0
    else:
        score = (pos_sim * 0.5 + colour_sim * 0.2 + fill_sim * 0.3) / total

    return [id_a, id_b, score]


def ref_compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
    """
    Compute a weighted similarity score between two 'patom' arrays.
    Assumes columns:
      0: patom_id (any hashable)
      3–4: normalized x,y positions
      5: colour
    Returns [id_a, id_b, similarity_score].
    """
    m, n = a.shape[0], b.shape[0]
    # Early exit on empties
    if m == 0 or n == 0:
        return [None, None, np.nan]

    # Extract and cast once to float32 (no full copy if already float32)
    pos1 = a[:, 3:5].astype(np.float32, copy=False)
    pos2 = b[:, 1:3].astype(np.float32, copy=False)
    col1 = a[:, 5].astype(np.float32, copy=False)
    col2 = b[:, 3].astype(np.float32, copy=False)

    # Sum of norms squared
    sum_sq1 = np.einsum('ij,ij->', pos1, pos1)      # ∑‖x_i‖²
    sum_sq2 = np.einsum('ij,ij->', pos2, pos2)      # ∑‖y_j‖²
    # Sum of all dot-products
    dot_sum = np.sum(pos1.dot(pos2.T))             # ∑_i ∑_j x_i·y_j

    # Total pairwise squared-distance sum
    total_sq = m * sum_sq2 + n * sum_sq1 - 2 * dot_sum
    pos_sim = np.sqrt(max(total_sq, 0.0)) / (m * n)

    # Colour similarity: sum |c_i - d_j| over all i,j
    colour_sim = np.abs(col1[:, None] - col2[None, :]).sum() / (m * n)

    # Pixel-count similarity
    fill_diff = abs(m - n) / ((m + n) / 2)
    fill_sim = min(fill_diff, 1.0)

    # IDs
    id_a, id_b = a[0, 0], b[0, 0]

    # Weighted combination
    total = pos_sim + colour_sim + fill_sim
    if total == 0.0:
        score = 0.0
    else:
        score = (pos_sim * 0.5 + colour_sim * 0.2 + fill_sim * 0.3) / total

    return [id_a, id_b, score]   