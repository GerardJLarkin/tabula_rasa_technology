import numpy as np
from typing import Any, List, Union

def compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
    # 4 columns (0, 1, 2, 3)
    # row 1 is id, centroid coordinates and segment
    # row 2 is min and max x and y values for original x and y coordinates in the frame
    # remaining rows are the normalised x and y values and the normalised colour at each coordinate

    # get normalise coordinates
    pos1 = a[2:, :2].astype(np.float32, copy=False)
    pos2 = b[2:, :2].astype(np.float32, copy=False)

    m, n = pos1.shape[0], pos2.shape[0]

    dists = np.linalg.norm(pos1[:,None,:] - pos2[None,:,:], axis=2)
    dists_norm = (dists - dists.min()) / (dists.max() - dists.min())
    pos_sim = dists_norm.mean()
    
    # get normalised colour values
    col1 = a[2:, 2].astype(np.float32, copy=False)
    col2 = b[2:, 2].astype(np.float32, copy=False)
    # Colour similarity: sum |c_i - d_j| over all i,j
    colour_sim = np.abs(col1[:, None] - col2[None, :]).sum() / (m * n)
    
    # Pixel-count similarity
    fill_diff = abs(m - n) / ((m + n) / 2)
    fill_sim = min(fill_diff, 1.0)

    # IDs
    id_a, id_b = str(a[0, 0]), str(b[0, 0])

    # weighted combination
    # total = pos_sim + colour_sim + fill_sim
    total = 7
    if total == 0.0:
        score = 0.0
        #print('exact', 0.0)
    elif (pos_sim <= 0.2) and (fill_sim <= 0.2):
        score = 0.2
        #print('pos fill', 0.2)
    else:
        score = ((pos_sim * 4) + (colour_sim * 1) + (fill_sim * 2)) / total
        #print('other', score)

    return [id_a, id_b, score]


def ref_compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
    # 4 columns (0, 1, 2, 3)
    # row 1 is id, centroid coordinates and segment
    # row 2 is min and max x and y values for original x and y coordinates in the frame
    # remaining rows are the normalised x and y values and the normalised colour at each coordinate

    m, n = a.shape[0], b.shape[0]
    # Early exit on empties
    if m == 0 or n == 0:
        return [None, None, np.nan]

    # Extract and cast once to float32 (no full copy if already float32)
    pos1 = a[2:, :2].astype(np.float32, copy=False)
    pos2 = b[2:, :2].astype(np.float32, copy=False)
    col1 = a[2:, 2].astype(np.float32, copy=False)
    col2 = b[2:, 2].astype(np.float32, copy=False)

    # get difference between points (reshape to deal with different sized arrays)
    diff = pos1[:, None, :] - pos2[None, :, :]
    # square the differences using einsum
    sq_distances = np.einsum('ijk,ijk->ij', diff, diff) 
    # get the square root for the euclidean distance
    pos_sim = np.sqrt(sq_distances).mean() / (m * n)

    # Colour similarity: sum |c_i - d_j| over all i,j
    colour_sim = np.abs(col1[:, None] - col2[None, :]).sum() / (m * n)
    
    # Pixel-count similarity
    fill_diff = abs(m - n) / ((m + n) / 2)
    fill_sim = min(fill_diff, 1.0)

    # IDs
    id_a, id_b = str(a[0, 0]), str(b[0, 0])

    # weighted combination
    total = pos_sim + colour_sim + fill_sim
    if total == 0.0:
        score = 0.0
    elif (pos_sim <= 0.3) and (fill_sim <= 0.3):
        score = 0.3
    else:
        score = (pos_sim * 0.1 + colour_sim * 0.6 + fill_sim * 0.3) / total

    return [id_a, id_b, score] 