# import numpy as np
# from typing import Any, List, Union

# from numba import njit

# @njit
# def compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
#     # 4 columns (0, 1, 2, 3)
#     # row 1 is id, centroid coordinates and segment
#     # row 2 is min and max x and y values for original x and y coordinates in the frame
#     # remaining rows are the normalised x and y values and the normalised colour at each coordinate
    
#     # IDs
#     id_a, id_b = a[0, 0], b[0, 0]

#     # get normalised coordinates
#     pos1 = a[2:, :2]
#     pos2 = b[2:, :2]

#     m, n = pos1.shape[0], pos2.shape[0]

#     # pixel-count similarity
#     fill_diff = abs(m - n) / ((m + n) / 2)
#     fill_sim = min(fill_diff, 1.0)
    
#     # adding in this heuristic check to attempt to improve processing efficiency - don't want to do it as the same shape (big vs small circle) can
#     # have very different number pixels
#     # (this does not significantly reduce the processing overhead when generating the reference patoms, therefore not being used)
#     # if fill_sim >= 0.5:
#     #     score = 1.0
#     # else:

#     diff = pos1[:, None, :] - pos2[None, :, :]  # Shape (m, n, 2)
#     dists = np.sum(diff ** 2, axis=-1)     # Shape (m, n), same as norm(axis=2)
    
#     dists_denom = dists.max() - dists.min()
#     adj_dists_denom = np.where(dists_denom == 0, 1, dists_denom)
#     dists_norm = (dists - dists.min()) / adj_dists_denom
#     pos_sim = dists_norm.mean()
    
#     # get normalised colour values
#     col1 = a[2:, 2]
#     col2 = b[2:, 2]
#     # colour similarity:
#     colour_sim = np.abs(col1[:, None] - col2[None, :]).mean() #.sum() / (m * n)

#     # weighted combination
#     if (pos_sim + colour_sim + fill_sim) == 0.0:
#         score = 0.0
#         #print('exact', 0.0)
#     elif (pos_sim <= 0.20) and (fill_sim <= 0.20):
#         score = 0.20
#         #print('pos fill', 0.2)
#     else:
#         score = ((pos_sim * 0.1) + (colour_sim * 0.6) + (fill_sim * 0.3))
#         #print('other', score)

#     return [id_a, id_b, score]

# import numpy as np
# from typing import Any, List, Union
# from numba import njit

# @njit
# def compute_radial_hist(coords: np.ndarray, bins: int = 50) -> np.ndarray:
#     centroid = np.mean(coords, axis=0)
#     distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
#     max_dist = np.max(distances)

#     if max_dist == 0:
#         hist = np.zeros(bins)
#         hist[0] = 1.0
#         return hist

#     hist = np.zeros(bins)
#     bin_edges = np.linspace(0, max_dist, bins + 1)
#     for d in distances:
#         for i in range(bins):
#             if bin_edges[i] <= d < bin_edges[i+1]:
#                 hist[i] += 1
#                 break
#     hist_sum = hist.sum()
#     if hist_sum > 0:
#         hist /= hist_sum
#     return hist

# @njit
# def radial_distribution_similarity(coords1: np.ndarray, coords2: np.ndarray, bins: int = 50) -> float:
#     hist1 = compute_radial_hist(coords1, bins)
#     hist2 = compute_radial_hist(coords2, bins)
#     similarity = np.minimum(hist1, hist2).sum()
#     return similarity

# @njit
# def compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
#     # 4 columns (0, 1, 2, 3)
#     # row 1 is id, centroid coordinates and segment
#     # row 2 is min and max x and y values for original x and y coordinates in the frame
#     # remaining rows are the normalised x and y values and the normalised colour at each coordinate

#     # IDs
#     id_a, id_b = a[0, 0], b[0, 0]

#     # get normalised coordinates
#     pos1 = a[2:, :2]
#     pos2 = b[2:, :2]

#     m, n = pos1.shape[0], pos2.shape[0]

#     # pixel-count similarity
#     fill_diff = abs(m - n) / ((m + n) / 2)
#     fill_sim = min(fill_diff, 1.0)

#     # Radial Distribution Similarity
#     radial_sim = 1 - radial_distribution_similarity(pos1, pos2)

#     # get normalised colour values
#     col1 = a[2:, 2]
#     col2 = b[2:, 2]
#     # colour similarity:
#     colour_sim = np.abs(col1[:, None] - col2[None, :]).mean() #.sum() / (m * n)

#     # weighted combination
#     if (radial_sim + colour_sim + fill_sim) == 0.0:
#         score = 0.0
#     elif (radial_sim >= 0.80) and (fill_sim <= 0.20):
#         score = 0.20
#     else:
#         score = ((radial_sim * 0.1) + (colour_sim * 0.6) + (fill_sim * 0.3))

#     return [id_a, id_b, score]

# # Note: pos_sim calculation has been replaced by radial distribution similarity.
# # For more precision, you can optionally hybrid-compare after a radial_sim threshold filter if desired.

# # Usage example:
# # result = compare(shape_array1, shape_array2)
# # shape_arrayX must be structured as per your original script with rows of id, coords, and colour data.

# import numpy as np
# from typing import Any, List, Union
# from numba import njit

# @njit
# def compute_radial_hist(coords: np.ndarray, bins: int = 20) -> np.ndarray:
#     centroid = np.mean(coords, axis=0)
#     distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
#     max_dist = np.max(distances)

#     if max_dist == 0:
#         hist = np.zeros(bins)
#         hist[0] = 1.0
#         return hist

#     hist = np.zeros(bins)
#     bin_edges = np.linspace(0, max_dist, bins + 1)
#     for d in distances:
#         for i in range(bins):
#             if bin_edges[i] <= d < bin_edges[i+1]:
#                 hist[i] += 1
#                 break
#     hist_sum = hist.sum()
#     if hist_sum > 0:
#         hist /= hist_sum
#     return hist

# @njit
# def compute_radial_colour_hist(coords: np.ndarray, colours: np.ndarray, bins: int = 20) -> np.ndarray:
#     centroid = np.mean(coords, axis=0)
#     distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
#     max_dist = np.max(distances)

#     if max_dist == 0:
#         hist = np.zeros(bins)
#         hist[0] = colours.mean()
#         return hist

#     hist = np.zeros(bins)
#     counts = np.zeros(bins)
#     bin_edges = np.linspace(0, max_dist, bins + 1)
#     for idx in range(distances.shape[0]):
#         d = distances[idx]
#         c = colours[idx]
#         for i in range(bins):
#             if bin_edges[i] <= d < bin_edges[i+1]:
#                 hist[i] += c
#                 counts[i] += 1
#                 break
#     for i in range(bins):
#         if counts[i] > 0:
#             hist[i] /= counts[i]
#     return hist

# @njit
# def radial_distribution_similarity(coords1: np.ndarray, coords2: np.ndarray, bins: int = 20) -> float:
#     hist1 = compute_radial_hist(coords1, bins)
#     hist2 = compute_radial_hist(coords2, bins)
#     similarity = np.minimum(hist1, hist2).sum()
#     return similarity

# @njit
# def radial_colour_similarity(coords1: np.ndarray, colours1: np.ndarray, coords2: np.ndarray, colours2: np.ndarray, bins: int = 20) -> float:
#     hist1 = compute_radial_colour_hist(coords1, colours1, bins)
#     hist2 = compute_radial_colour_hist(coords2, colours2, bins)
#     diff = np.abs(hist1 - hist2)
#     similarity = 1.0 - diff.mean()  # Similarity as inverse of mean absolute difference
#     return similarity

# @njit
# def compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
#     # 4 columns (0, 1, 2, 3)
#     # row 1 is id, centroid coordinates and segment
#     # row 2 is min and max x and y values for original x and y coordinates in the frame
#     # remaining rows are the normalised x and y values and the normalised colour at each coordinate

#     # IDs
#     id_a, id_b = a[0, 0], b[0, 0]

#     # get normalised coordinates and colours
#     pos1 = a[2:, :2]
#     pos2 = b[2:, :2]
#     col1 = a[2:, 2]
#     col2 = b[2:, 2]

#     m, n = pos1.shape[0], pos2.shape[0]

#     # pixel-count similarity
#     fill_diff = abs(m - n) / ((m + n) / 2)
#     fill_sim = min(fill_diff, 1.0)

#     # Radial Distribution Similarity (Position)
#     radial_sim = radial_distribution_similarity(pos1, pos2)

#     # Radial Colour Distribution Similarity
#     radial_colour_sim = radial_colour_similarity(pos1, col1, pos2, col2)

#     # weighted combination
#     if (radial_sim + radial_colour_sim + fill_sim) == 0.0:
#         score = 0.0
#     elif (radial_sim >= 0.80) and (fill_sim <= 0.20):
#         score = 0.20
#     else:
#         score = ((radial_sim * 0.1) + (radial_colour_sim * 0.6) + (fill_sim * 0.3))

#     return [id_a, id_b, score]

# # Note: Position and Colour are now both compared using radial distribution profiles.
# # Usage example:
# # result = compare(shape_array1, shape_array2)
# # shape_arrayX must be structured as per your original script with rows of id, coords, and colour data.


# import numpy as np
# from typing import Any, List, Union
# from numba import njit

# @njit
# def compute_radial_hist(coords: np.ndarray, bins: int = 20) -> np.ndarray:
#     centroid = np.mean(coords, axis=0)
#     distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
#     max_dist = np.max(distances)

#     if max_dist == 0:
#         hist = np.zeros(bins)
#         hist[0] = 1.0
#         return hist

#     hist = np.zeros(bins)
#     bin_edges = np.linspace(0, max_dist, bins + 1)
#     for d in distances:
#         for i in range(bins):
#             if bin_edges[i] <= d < bin_edges[i+1]:
#                 hist[i] += 1
#                 break
#     hist_sum = hist.sum()
#     if hist_sum > 0:
#         hist /= hist_sum
#     return hist

# @njit
# def compute_radial_colour_hist(coords: np.ndarray, colours: np.ndarray, bins: int = 20) -> np.ndarray:
#     centroid = np.mean(coords, axis=0)
#     distances = np.sqrt(((coords - centroid) ** 2).sum(axis=1))
#     max_dist = np.max(distances)

#     if max_dist == 0:
#         hist = np.zeros(bins)
#         hist[0] = colours.mean()
#         return hist

#     hist = np.zeros(bins)
#     counts = np.zeros(bins)
#     bin_edges = np.linspace(0, max_dist, bins + 1)
#     for idx in range(distances.shape[0]):
#         d = distances[idx]
#         c = colours[idx]
#         for i in range(bins):
#             if bin_edges[i] <= d < bin_edges[i+1]:
#                 hist[i] += c
#                 counts[i] += 1
#                 break
#     for i in range(bins):
#         if counts[i] > 0:
#             hist[i] /= counts[i]
#     return hist

# @njit
# def radial_distribution_similarity(coords1: np.ndarray, coords2: np.ndarray, bins: int = 20) -> float:
#     hist1 = compute_radial_hist(coords1, bins)
#     hist2 = compute_radial_hist(coords2, bins)
#     similarity = np.minimum(hist1, hist2).sum()
#     return similarity

# @njit
# def radial_colour_similarity(coords1: np.ndarray, colours1: np.ndarray, coords2: np.ndarray, colours2: np.ndarray, bins: int = 20) -> float:
#     hist1 = compute_radial_colour_hist(coords1, colours1, bins)
#     hist2 = compute_radial_colour_hist(coords2, colours2, bins)
#     diff = np.abs(hist1 - hist2)
#     similarity = 1.0 - diff.mean()  # Similarity as inverse of mean absolute difference
#     return similarity

# @njit
# def segment_fill_similarity(coords1: np.ndarray, coords2: np.ndarray, angular_bins: int = 16) -> float:
#     centroid1 = np.mean(coords1, axis=0)
#     centroid2 = np.mean(coords2, axis=0)

#     vectors1 = coords1 - centroid1
#     vectors2 = coords2 - centroid2

#     angles1 = np.arctan2(vectors1[:, 1], vectors1[:, 0])
#     angles2 = np.arctan2(vectors2[:, 1], vectors2[:, 0])

#     angles1 = (angles1 + 2 * np.pi) % (2 * np.pi)
#     angles2 = (angles2 + 2 * np.pi) % (2 * np.pi)

#     hist1 = np.zeros(angular_bins)
#     hist2 = np.zeros(angular_bins)

#     bin_edges = np.linspace(0, 2 * np.pi, angular_bins + 1)

#     for angle in angles1:
#         for i in range(angular_bins):
#             if bin_edges[i] <= angle < bin_edges[i+1]:
#                 hist1[i] += 1
#                 break

#     for angle in angles2:
#         for i in range(angular_bins):
#             if bin_edges[i] <= angle < bin_edges[i+1]:
#                 hist2[i] += 1
#                 break

#     hist1 /= hist1.sum()
#     hist2 /= hist2.sum()

#     diff = np.abs(hist1 - hist2)
#     similarity = 1.0 - diff.mean()
#     return similarity

# @njit
# def compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
#     # 4 columns (0, 1, 2, 3)
#     # row 1 is id, centroid coordinates and segment
#     # row 2 is min and max x and y values for original x and y coordinates in the frame
#     # remaining rows are the normalised x and y values and the normalised colour at each coordinate

#     # IDs
#     id_a, id_b = a[0, 0], b[0, 0]

#     # get normalised coordinates and colours
#     pos1 = a[2:, :2]
#     pos2 = b[2:, :2]
#     col1 = a[2:, 2]
#     col2 = b[2:, 2]

#     m, n = pos1.shape[0], pos2.shape[0]

#     # pixel-count similarity
#     fill_diff = abs(m - n) / ((m + n) / 2)
#     fill_sim = min(fill_diff, 1.0)

#     # Radial Distribution Similarity (Position)
#     radial_sim = radial_distribution_similarity(pos1, pos2)

#     # Radial Colour Distribution Similarity
#     radial_colour_sim = radial_colour_similarity(pos1, col1, pos2, col2)

#     # Segment Fill Similarity (Rotation Invariant)
#     segment_fill_sim = segment_fill_similarity(pos1, pos2)

#     # weighted combination
#     score = (radial_sim * 0.1) + (radial_colour_sim * 0.5) + (fill_sim * 0.2) + (segment_fill_sim * 0.2)

#     return [id_a, id_b, score]

# Note: Added segment-based fill similarity which is rotation invariant.
# Usage example:
# result = compare(shape_array1, shape_array2)
# shape_arrayX must be structured as per your original script with rows of id, coords, and colour data.


import numpy as np
from typing import Any, List, Union

def compute_radial_hist(coords: np.ndarray, bins: int = 50) -> np.ndarray:
    centroid = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    max_dist = distances.max()

    if max_dist == 0:
        hist = np.zeros(bins)
        hist[0] = 1.0
        return hist

    hist = np.histogram(distances, bins=bins, range=(0, max_dist))[0]
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    return hist

def compute_radial_colour_hist(coords: np.ndarray, colours: np.ndarray, bins: int = 50) -> np.ndarray:
    centroid = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    max_dist = distances.max()

    if max_dist == 0:
        hist = np.zeros(bins)
        hist[0] = colours.mean()
        return hist

    bin_indices = np.digitize(distances, bins=np.linspace(0, max_dist, bins + 1)) - 1
    hist = np.zeros(bins)
    counts = np.zeros(bins)
    for idx, bin_idx in enumerate(bin_indices):
        if 0 <= bin_idx < bins:
            hist[bin_idx] += colours[idx]
            counts[bin_idx] += 1
    counts = np.where(counts == 0, 1, counts)
    hist /= counts
    return hist

def segment_fill_histogram(coords: np.ndarray, angular_bins: int = 16) -> np.ndarray:
    centroid = np.mean(coords, axis=0)
    vectors = coords - centroid
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    hist = np.histogram(angles, bins=angular_bins, range=(0, 2 * np.pi))[0]
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    return hist

def compare(shapes: List[np.ndarray], bins: int = 50, angular_bins: int = 16) -> List[List[Union[int, int, float]]]:
    descriptors = []
    for shape in shapes:
        pos = shape[2:, :2]
        col = shape[2:, 2]

        #fill_count = pos.shape[0]
        radial_hist = compute_radial_hist(pos, bins)
        radial_colour_hist = compute_radial_colour_hist(pos, col, bins)
        segment_hist = segment_fill_histogram(pos, angular_bins)

        descriptors.append({
            'id': shape[0, 0],
            #'fill_count': fill_count,
            'radial_hist': radial_hist,
            'radial_colour_hist': radial_colour_hist,
            'segment_hist': segment_hist
        })

    comparisons = []
    for i in range(len(descriptors)):
        for j in range(i + 1, len(descriptors)):
            desc1 = descriptors[i]
            desc2 = descriptors[j]

            # Fill similarity
            # m, n = desc1['fill_count'], desc2['fill_count']
            # fill_diff = abs(m - n) / ((m + n) / 2)
            # fill_sim = min(fill_diff, 1.0)

            # Radial Position Similarity
            radial_sim = np.minimum(desc1['radial_hist'], desc2['radial_hist']).sum()

            # Radial Colour Similarity
            diff = np.abs(desc1['radial_colour_hist'] - desc2['radial_colour_hist']).mean()
            radial_colour_sim = 1.0 - diff

            # Segment Fill Similarity
            segment_diff = np.abs(desc1['segment_hist'] - desc2['segment_hist']).mean()
            segment_fill_sim = 1.0 - segment_diff

            # Weighted Score
            score = (radial_sim * 0.1) + (radial_colour_sim * 0.6) + (segment_fill_sim * 0.3) # (fill_sim * 0.2) +

            comparisons.append([desc1['id'], desc2['id'], score])

    return comparisons

# Usage example:
# results = compare_shapes_batch([shape_array1, shape_array2, shape_array3])
# Each shape_arrayX must be structured as per your original script with rows of id, coords, and colour data.

