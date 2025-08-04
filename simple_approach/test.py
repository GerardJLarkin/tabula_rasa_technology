# from typing import Callable
# import numpy as np

# class ArrayGroup:
#     """
#     Holds one reference array + a count of how many arrays contributed to it.
#     On each add, updates reference via inc_ref_fn(current_ref, new_arr, count).
#     """
#     def __init__(
#         self,
#         initial_array: np.ndarray,
#         inc_ref_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
#     ):
#         self.reference = np.array(initial_array, copy=True)
#         self.inc_ref_fn = inc_ref_fn
#         self.count = 1

#     def add_member(self, arr: np.ndarray):
#         arr = np.array(arr, copy=False)
#         # compute new reference from the old reference, new array, and how many we've seen
#         self.reference = self.inc_ref_fn(self.reference, arr, self.count)
#         self.count += 1

#     def as_list(self):
#         """Return [reference_array, count]"""
#         return [self.reference, self.count]


# class ArrayGroupManager:
#     """
#     Maintains multiple ArrayGroup instances.
#     On add_array:
#       - compares the new array to each group's reference
#       - if best similarity ≥ threshold: adds to that group (and updates its reference & count)
#       - otherwise makes a new group with count=1 and reference=new array
#     """
#     def __init__(
#         self,
#         compare_fn: Callable[[np.ndarray, np.ndarray], float],
#         inc_ref_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
#         threshold: float
#     ):
#         self.compare_fn = compare_fn
#         self.inc_ref_fn = inc_ref_fn
#         self.threshold = threshold
#         self.groups: list[ArrayGroup] = []

#     def add_array(self, arr: np.ndarray):
#         arr = np.array(arr, copy=False)
#         best_group = None
#         best_sim = -np.inf

#         # find the most-similar existing reference
#         for group in self.groups:
#             sim = self.compare_fn(group.reference, arr)
#             if sim > best_sim:
#                 best_sim, best_group = sim, group

#         # decide whether to join or start new
#         if best_group is None or best_sim < self.threshold:
#             # new reference group
#             new_group = ArrayGroup(arr, self.inc_ref_fn)
#             self.groups.append(new_group)
#         else:
#             best_group.add_member(arr)

#     def get_all_groups(self):
#         """
#         Returns a list of [reference_array, count] for each group.
#         """
#         return [g.as_list() for g in self.groups]


# # 1) your similarity function: ref, arr → float
# def my_compare(a: np.ndarray, b: np.ndarray) -> float:
#     # e.g. cosine sim
#     return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

# # 2) your incremental reference updater: (old_ref, new_arr, count) → updated_ref
# def my_inc_ref(old_ref: np.ndarray, new_arr: np.ndarray, count: int) -> np.ndarray:
#     # example = running mean:
#     # new_ref = (old_ref * count + new_arr) / (count + 1)
#     return (old_ref * count + new_arr) / (count + 1)

# # 3) build the manager
# mgr = ArrayGroupManager(
#     compare_fn=my_compare,
#     inc_ref_fn=my_inc_ref,
#     threshold=0.25
# )

# # 4) stream in your arrays
# for new_arr in stream_of_new_arrays:
#     mgr.add_array(new_arr)

# # 5) inspect
# for ref, cnt in mgr.get_all_groups():
#     print("Reference shape:", ref.shape, "Count:", cnt)

import numpy as np
import os, glob
from time import perf_counter
import sys
from itertools import combinations

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(root, 'historic_data')

from tabula_rasa_technology.simple_approach.compare import compare

arrays = [np.load(i) for i in glob.glob(os.path.join(folder, '*.npy'))]


print(sys.getsizeof(arrays))


# Example list of numpy arrays


# Threshold for similarity (e.g., Euclidean distance)
threshold = 0.25

# List to store groups of similar arrays
groups = []

# List to keep track of arrays already grouped
used_indices = set()

# Loop through each array in the list
for i in range(len(arrays)):
    if i in used_indices:
        continue  # Skip if already grouped

    current_group = [arrays[i]]
    used_indices.add(i)

    # Compare current array with every other array
    for j in range(i + 1, len(arrays)):
        if j in used_indices:
            continue

        # Example comparison: Euclidean distance
        distance = np.linalg.norm(arrays[i] - arrays[j])

        if distance < threshold:
            current_group.append(arrays[j])
            used_indices.add(j)

    # Add the group (whether single or multiple arrays) to groups list
    groups.append(current_group)

# Print grouped arrays
for idx, group in enumerate(groups):
    print(f"Group {idx + 1}:")
    for arr in group:
        print(arr)
    print("-----")
