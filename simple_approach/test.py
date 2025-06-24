# import os
# import numpy as np
# from itertools import combinations
# from multiprocessing import Pool

# # 1. Optimized compare
# def compare_optimized(a: np.ndarray, b: np.ndarray):
#     m, n = a.shape[0], b.shape[0]
#     # Handle empty segments as maximally dissimilar
#     if m == 0 or n == 0:
#         return [int(a[0,0]) if m else None,
#                 int(b[0,0]) if n else None,
#                 np.inf]

#     sum_ax, sum_ay = a[:,3].sum(), a[:,4].sum()
#     sum_bx, sum_by = b[:,3].sum(), b[:,4].sum()

#     dx_total = sum_bx * m - sum_ax * n
#     dy_total = sum_by * m - sum_ay * n
#     position_similarity = np.hypot(dx_total, dy_total) / (m * n)

#     colour_similarity = abs(a[:,5].sum() * n - b[:,5].sum() * m) / (m * n)

#     pixel_fill_similarity = abs(m - n) / ((m + n) / 2)

#     score = 0.4 * position_similarity + \
#             0.2 * colour_similarity + \
#             0.4 * pixel_fill_similarity

#     return [int(a[0,0]), int(b[0,0]), score]


# # 2. Union-Find for grouping
# class UnionFind:
#     def __init__(self):
#         self.parent = {}
#     def find(self, x):
#         if self.parent.setdefault(x, x) != x:
#             self.parent[x] = self.find(self.parent[x])
#         return self.parent[x]
#     def union(self, x, y):
#         rx, ry = self.find(x), self.find(y)
#         if rx != ry:
#             self.parent[ry] = rx


# # 3. Load all patoms into memory once
# def load_patoms(dirpath):
#     patoms = {}
#     for fname in os.listdir(dirpath):
#         if not fname.endswith('.npy'):
#             continue
#         arr = np.load(os.path.join(dirpath, fname))
#         patom_id = int(arr[0,0])
#         patoms[patom_id] = arr
#     return patoms


# # 4. Find all similar pairs in one batched starmap
# def find_similar_pairs(patoms, threshold, n_procs=None):
#     ids = list(patoms.keys())
#     arrays = [patoms[i] for i in ids]

#     # Build all index‐pairs
#     idx_pairs = [(i, j) for i in range(len(ids)) for j in range(i+1, len(ids))]

#     # Map back to actual data and compare
#     with Pool(processes=n_procs) as pool:
#         tasks = ((arrays[i], arrays[j]) for i,j in idx_pairs)
#         results = pool.starmap(compare_optimized, tasks, chunksize=500)

#     # Filter by threshold
#     return [(one, two) for one, two, score in results if score < threshold]


# # 5. Group via Union-Find
# def group_ids(pairs):
#     uf = UnionFind()
#     for a, b in pairs:
#         uf.union(a, b)
#     groups = {}
#     for node in uf.parent:
#         root = uf.find(node)
#         groups.setdefault(root, []).append(node)
#     return list(groups.values())


# # 6. Build reference patoms with vectorized stats
# def build_reference_patoms(patoms, groups):
#     refs = {}
#     for grp in groups:
#         # Stack all arrays in this group
#         data = np.vstack([patoms[i] for i in grp])
#         # Compute mean & median across rows
#         means   = np.mean(data,   axis=0)
#         medians = np.median(data, axis=0)
#         refs[grp[0]] = {'mean': means, 'median': medians}
#     return refs


# # 7. Main orchestration
# def main():
#     datadir      = '/path/to/your/patoms'
#     sim_threshold = 0.3
#     patoms       = load_patoms(datadir)
#     pairs        = find_similar_pairs(patoms, sim_threshold, n_procs=8)
#     groups       = group_ids(pairs)
#     references   = build_reference_patoms(patoms, groups)

#     # (Save or return `references` as you need)
#     print(f"Found {len(groups)} groups, built {len(references)} reference patoms.")

# if __name__ == '__main__':
#     main()



# import os
# import numpy as np

# def compute_reference_for_group(group_arrays):
#     """
#     Your original reference‐patom logic goes here,
#     e.g. looping over columns to pick means, medians, modes, etc.
#     Must return a NumPy array `ref_array` of shape (…,7).
#     """
#     # … your existing implementation …
#     return ref_array


# def write_reference_patoms(patoms, groups, out_dir):
#     """
#     For each connected group of patom IDs:
#       1. Gather their arrays in memory (you already have `patoms` dict).
#       2. Compute your reference array with the original method.
#       3. Save that single reference array to disk with np.save.
#       4. Delete the array to free memory before next group.
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     for grp in groups:
#         # 1. load the group's arrays (already in memory)
#         group_arrays = [patoms[i] for i in grp]

#         # 2. compute reference using your original code
#         ref_array = compute_reference_for_group(group_arrays)

#         # 3. save to disk
#         #    e.g. use the first patom ID as filename
#         fname = f"ref_patoms_{grp[0]}.npy"
#         path  = os.path.join(out_dir, fname)
#         np.save(path, ref_array)

#         # 4. free memory
#         del group_arrays, ref_array

#     print(f"Wrote {len(groups)} reference files to {out_dir}")


#

# intelligence building
import numpy as np
import os
import glob
import sys
from time import perf_counter
from typing import Iterable, Callable, Generator, Any, Tuple
from typing import Callable, List, Tuple, Set
from itertools import product
import pickle
import math
import random
import string

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')

from simple_approach_compare_v2 import ref_compare
from simple_approach_patoms_v1 import patoms

root = os.path.dirname(os.path.abspath(__file__))
# load reference patoms
# how will this impact on memory? can I hold them all and still save room for processing?
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]
print('reference loaded')

ref_ids = [i[0,0] for i in ref_patoms]
segments = [f"{i[0]:0>2}"+f"{i[1]:0>2}" for i in product((range(16)), repeat=2)]
vec_keys = [i+j for i, j in product(ref_ids, segments)]
print(vec_keys[:10])

vrld = dict.fromkeys(vec_keys, 0.0) # 15.4MB to start

print(sys.getsizeof(vrld))

# # Using the same input dataset as per the compartor CNN-LSTM model
# data = np.load('mnist_test_seq.npy')
# # Swap the axes representing the number of frames and number of data samples.
# dataset = np.swapaxes(data, 0, 1)
# # We'll pick out 1000 of the 10000 total examples and use those.
# dataset = dataset[:100, ...]
# print('loaded')
# #generate patoms from sequences and save to disk
# seq_ind = 0
# for i in range(0,1,1):
#     print('sequnce num:',i)
#     sequence = dataset[i]
#     for j in range(0,20,1):
#         print(sequence[j].shape)


frame_size = np.zeros((64,64))

test_data = [] # centroind values will change
for i in range(6):
    patom_id = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(6))
    x_cent = np.random.randint(low=0, high=63, dtype='int')
    y_cent = np.random.randint(low=0, high=63, dtype='int')
    id_pat_centroid = (patom_id, x_cent, y_cent)
    test_data.append(id_pat_centroid)

print(test_data)
center_x = (frame_size.shape[0]-1)/2; center_y = (frame_size.shape[1]-1)/2 # scalar will never change
num_segments = 16 # scalar will never change
segment_width = 360 / num_segments
print(center_x, center_y)
initial_segments = []
for i in test_data:
    x = i[1]; y = i[2]
    angle_deg = (np.degrees(np.arctan2(center_y - y, x - center_x)) + 360) % 360
    angle_clockwise_from_north = (90 - angle_deg) % 360
    segment = f"{(angle_clockwise_from_north // segment_width).astype(int):0>2}"
    initial_segments.append(segment)

print(initial_segments)

# def find_vector(data):
#     for a, b, c in zip((64,64), data, data[1:], data[2:]):
#         segment = 360 / 16
#         center_x = math.floor((64-1)/2); center_y = math.floor((64-1)/2)
#         strt_dx1 = a[1] - center_x
#         strt_dy1 = center_y - a[2]
#         start_seg_1 = f"{int((np.degrees(np.arctan2(strt_dx1, strt_dy1)) % 360) // segment):0>2}"
#         dx1 = b[1] - a[1]
#         dy1 = b[2] - a[2]
#         seg_1 = f"{int((np.degrees(np.arctan2(dy1, dx1)) % 360) // segment):0>2}"
#         start_seg_2 = f"{int((np.degrees(np.arctan2(b[1], b[2])) % 360) // segment):0>2}"
#         dx2 = c[1] - b[1]
#         dy2 = c[2] - b[2]
#         seg_2 = f"{int((np.degrees(np.arctan2(dy2, dx2)) % 360) // segment):0>2}"
#         print(f"F2F vector direction: {a[0]}→{b[0]} = {start_seg_1+seg_1}")
#         print(f"F2F vector direction: {b[0]}→{c[0]} = {start_seg_2+seg_2}")


# find_vector(test_data)