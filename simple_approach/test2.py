# import itertools

# # your inputs
# ids1 = ["A", "B", "C"]
# ids2 = ["X", "Y"]
# directions = [f"{i*15}°" for i in range(24)]
# nest_dict = {'f': 0.0, 'v': (0.0, 0.0)}

# # build the nested dict
# nested = {
#     f"{i1}-{i2}": nest_dict
#     for i1, i2 in itertools.product(ids1, ids2)
# }

# # inspect one entry
# key = "A-X"
# print(key, "→", len(nested[key]), "direction‐pairs")
# print(nested)  # first five pairs

# def reverse_normalize(value, center, range_min=0, range_max=64):
#     """
#     Reverses the normalization of a value between -1 and 1 to the original range.

#     Args:
#         value: The normalized value (-1 to 1).
#         center: The center point within the original range.
#         range_min: The minimum value of the original range (default 0).
#         range_max: The maximum value of the original range (default 64).

#     Returns:
#         The reversed normalized value in the original range.
#     """
#     if not -1 <= value <= 1:
#         raise ValueError("Normalized value must be between -1 and 1")

#     range_width = range_max - range_min
#     # Scale the normalized value to the full range width.
#     scaled_value = value * (range_width / 2)
#     # Shift the scaled value to the correct position relative to the center
#     original_value = center + scaled_value

#     return original_value


# # Example usage:
# normalized_value = 0.5  # Example normalized value between -1 and 1
# center_point = 20  # Example center point
# original_value = reverse_normalize(normalized_value, center_point)
# print(f"Reversed normalized value: {original_value}")  # Output will be 42.0
# #Example with a negative normalized value
# normalized_value = -0.75
# original_value = reverse_normalize(normalized_value, center_point)
# print(f"Reversed normalized value: {original_value}") #output will be 8.0

# #Example within the range limits
# normalized_value = 1.0
# original_value = reverse_normalize(normalized_value, center_point)
# print(f"Reversed normalized value: {original_value}") #output will be 52.0

# normalized_value = -1.0
# original_value = reverse_normalize(normalized_value, center_point)
# print(f"Reversed normalized value: {original_value}") #output will be 12.0


import numpy as np
from typing import List

def merge_point_lists(
    point_lists: List[np.ndarray],
    shape: tuple[int, int],
    mode: str = "sum"  # or "overwrite"
) -> np.ndarray:
    """
    Given a list of N×3 arrays, each with columns [x, y, value],
    returns a single 2D array of shape `shape` filled with the
    values from all lists at their (x,y) coords.

    If mode=="sum", overlapping indices are summed;
    if mode=="overwrite", later lists simply overwrite earlier.
    """
    result = np.zeros(shape, dtype=float)

    for pts in point_lists:
        # pts[:,0] = x indices, pts[:,1] = y indices, pts[:,2] = values
        xs = pts[:, 0].astype(int)
        ys = pts[:, 1].astype(int)
        vals = pts[:, 2]

        if mode == "sum":
            # accumulate (works even if xs/ys contain duplicates)
            np.add.at(result, (ys, xs), vals)
        else:  # overwrite
            result[ys, xs] = vals

    return result

# ── Example ────────────────────────────────────────────────────────────────
# Suppose you have three small “point” arrays:
A = np.array([
    [0, 0, 1.0],
    [1, 2, 2.5],
])
print(A.shape)
B = np.array([
    [1, 2, 3.0],  # overlaps (1,2)
    [4, 5, 4.2],
])
C = np.array([
    [0, 0, 0.5],  # overlaps (0,0)
    [3, 3, 1.1],
])
point_lists = [A, B, C]

# choose image shape (must be large enough to include all x,y)
shape = (6, 6)

# a) sum mode: overlapping points get added
summed = merge_point_lists(point_lists, shape, mode="sum")
# summed[0,0] == 1.0 + 0.5 = 1.5
# summed[2,1] == 2.5 + 3.0 = 5.5

# b) overwrite mode: C’s 0,0 (0.5) overwrites A’s 1.0, etc.
overwritten = merge_point_lists(point_lists, shape, mode="overwrite")
# overwritten[0,0] == 0.5
# overwritten[2,1] == 3.0

print("Summed result:\n", summed)
print("Overwritten result:\n", overwritten)

