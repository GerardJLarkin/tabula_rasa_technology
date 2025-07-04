# create reference patoms
import numpy as np
import os
import glob
import sys
import random
import string
from itertools import islice
from multiprocessing import Pool
from time import perf_counter

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(root, 'historic_data')

from tabula_rasa_technology.simple_approach.compare_v1 import compare

# Getting all the numpy arrays .npy files based on matching pattern (*.npy)
file_paths = glob.glob(os.path.join(folder, '*.npy'))
# Import arrays from folder and store them as a dict
patoms = {}
for fname in os.listdir(folder):
    if not fname.endswith('.npy'):
        continue
    arr = np.load(os.path.join(folder, fname), allow_pickle=True)
    patom_id = str(arr[0,0])
    patoms[patom_id] = arr
print('historic data loaded')

## memory issue here - can't seem to get total number of patoms from dataset into memory and be useful
#patoms = dict(islice(patoms.items(), 8100))

ids = list(patoms.keys())
arrays = [patoms[i] for i in ids]
idx_pairs = [(i, j) for i in range(len(ids)) for j in range(i+1, len(ids))]

# set a similarity threshold
sim_threshold = 0.3
s = perf_counter()
with Pool(processes=4) as pool:
    tasks = ((arrays[i], arrays[j]) for i,j in idx_pairs)
    results = pool.starmap(compare, tasks, chunksize=1000)

    similar_patoms = [(one, two) for one, two, score in results if score <= sim_threshold]

# check which patoms have not been added to the similar patoms lists
not_similar_patoms = []
for i in ids:
    if not any(i in sublist for sublist in similar_patoms):
        not_similar_patoms.append(i)

# create a reference patom for each of the patom ids in the not similar list
# 4 columns (0, 1, 2, 3)
# row 1 is id, centroid coordinates and segment
# row 2 is min and max x and y values for original x and y coordinates in the frame
# remaining rows are the normalised x and y values and the normalised colour at each coordinate

# get non similar patoms from dictionary of patoms
non_sim_patom = [patoms[i] for i in not_similar_patoms]
# select only required columns
for pat in non_sim_patom:
    np.save(f'reference_patoms/patom_{str(pat[0,0])}', pat)

class UnionFind:
    def __init__(self):
        # maps each node → its parent in the forest
        self.parent = {}

    def find(self, x):
        # initialize x’s parent to itself if unseen
        if self.parent.setdefault(x, x) != x:
            # path-compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # attach one root under the other
            self.parent[ry] = rx


def extract_groups(pairs):
    """
    pairs: list of (id1, id2)
    returns: list of connected components (each a list of ids)
    """
    uf = UnionFind()
    # 1) Merge each pair
    for a, b in pairs:
        uf.union(a, b)

    # 2) Gather nodes under their ultimate root
    groups = {}
    for node in uf.parent:
        root = uf.find(node)
        groups.setdefault(root, []).append(node)

    return list(groups.values())

groups = extract_groups(similar_patoms)

for group in groups:
    group_patoms = [patoms[id] for id in group]
    num_patoms = len(group_patoms)

    # stack patoms to create a singluar group patom numpy array
    group_patoms = tuple(group_patoms)
    group_patoms = np.vstack(group_patoms)
    # i'm ok with average number of rows for now
    avg_rows = int(np.ceil(group_patoms.shape[0] / num_patoms))
    # 11 columns (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    # (patom_id, min_x, max_x, min_y, max_y, norm_x, norm_y, colours, x_cent, y_cent, segment)
    # ref patoms structure: ref_patom_id, min_x, max_x, min_y, max_y, norm_x, norm_y, colour
    
    x_vals, x_val_count = np.unique(group_patoms[:,5], return_counts=True)
    x_vals = x_vals.reshape(x_vals.shape[0],1); x_val_count = x_val_count.reshape(x_val_count.shape[0],1)
    x_vals = np.hstack((x_vals, x_val_count))
    x_desc_order = x_vals[:,-1].argsort()[::-1]
    x_vals_sorted = x_vals[x_desc_order]
    
    # if the avg num rows is smaller or equal to the shape of the x values and their counts take top rows up to avg row number
    if avg_rows <= x_vals_sorted.shape[0]:
        x_vals = x_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
    # if the avg num rows is greater than the shape of the x values and their counts expand the top n rows where the sum of the counts
    # related to each top n x value is equal to the avg num rows
    else:
        cumsum = np.cumsum(x_vals_sorted[:,1]).reshape(x_vals_sorted.shape[0],1)
        x_vals_cumsum = np.hstack((x_vals_sorted, cumsum))
        counts = x_vals_cumsum[:, 1].astype(int)
        expanded_x_vals_array = np.repeat(x_vals_cumsum, counts, axis=0)
        x_vals = expanded_x_vals_array[:avg_rows,0].reshape(avg_rows,1)

    # get avg min, max x 
    group_min_x = np.unique(group_patoms[:,1]).mean()
    group_max_x = np.unique(group_patoms[:,2]).mean()  
    min_x_arr = np.array([group_min_x] * avg_rows).reshape(avg_rows,1)
    max_x_arr = np.array([group_max_x] * avg_rows).reshape(avg_rows,1)

    y_vals, y_val_count = np.unique(group_patoms[:,6], return_counts=True)
    y_vals = y_vals.reshape(y_vals.shape[0],1); y_val_count = y_val_count.reshape(y_val_count.shape[0],1)
    y_vals = np.hstack((y_vals, y_val_count))
    y_desc_order = y_vals[:,-1].argsort()[::-1]
    y_vals_sorted = y_vals[y_desc_order]
    if avg_rows <= y_vals_sorted.shape[0]:
        y_vals = y_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
    else:
        cumsum = np.cumsum(y_vals_sorted[:,1]).reshape(y_vals_sorted.shape[0],1)
        y_vals_cumsum = np.hstack((y_vals_sorted, cumsum))
        counts = y_vals_cumsum[:, 1].astype(int)
        expanded_y_vals_array = np.repeat(y_vals_cumsum, counts, axis=0)
        y_vals = expanded_y_vals_array[:avg_rows,0].reshape(avg_rows,1)
    
    # get avg min, max y
    group_min_y = np.unique(group_patoms[:,3]).mean()
    group_max_y = np.unique(group_patoms[:,4]).mean()  
    min_y_arr = np.array([group_min_y] * avg_rows).reshape(avg_rows,1)
    max_y_arr = np.array([group_max_y] * avg_rows).reshape(avg_rows,1)

    x_y = np.hstack((min_x_arr, max_x_arr, min_y_arr, max_y_arr, x_vals, y_vals))
    
    #get 'average' colour at x,y postion?????
    # back to original vstacked group of patoms, extract pixel colours for each of the x values that made it in to the final cut
    x_colours = []
    for i in x_y[:,4].tolist():
        colours = group_patoms[:,7][group_patoms[:,5] == i]
        # get mode, mean and median
        mode_colour, colour_count = np.unique(colours, return_counts=True)
        mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
        mode_colour = np.hstack((mode_colour, colour_count))
        mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
        mode_color = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0,0]
        mean_colour = colours.mean()
        median_colour = np.median(colours)
        colour = (mode_colour + mean_colour + median_colour) / 3
        x_colours.append(colour)

    y_colours = []
    for i in x_y[:,5].tolist():
        colours = group_patoms[:,7][group_patoms[:,6] == i]
        # get mode, mean and median
        mode_colour, colour_count = np.unique(colours, return_counts=True)
        mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
        mode_colour = np.hstack((mode_colour, colour_count))
        mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
        mode_color = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0,0]
        mean_colour = colours.mean()
        median_colour = np.median(colours)
        colour = (mode_colour + mean_colour + median_colour) / 3
        y_colours.append(colour)

    x_y_colours = list(zip(x_colours, y_colours))
    x_y_colours = np.array([sum(i)/2 for i in x_y_colours]).reshape(avg_rows,1)

    # create a reference patom id
    ref_patom_id = np.random.default_rng().random(dtype=np.float32)
    ref_patom_id_arr = np.array([ref_patom_id] * avg_rows).reshape(avg_rows,1).astype('object')
    
    ref_patom = np.hstack((ref_patom_id_arr, x_y, x_y_colours))
    
    np.save(f'reference_patoms/patom_{str(ref_patom_id)}', ref_patom)

end = perf_counter()
print("Time taken (mins):", (end - start)/60)