# script to test the output of the patom generation script
import numpy as np

import sys

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.unnormalise import unnormalise_xy

# nearest neighbour offsets
motion = np.array([[-1,  1], [ 0,  1], [ 1,  1],
                   [-1,  0],           [ 1,  0],
                   [-1, -1], [ 0, -1], [ 1, -1]], dtype=np.int32)

threshold_val = 40.0

def all_nn_edges(arr, threshold = threshold_val):
    
    height, width = arr.shape
    orig_array = arr.astype(np.float32, copy=False)

    # pad array to allow for nearest neighbour comparison
    padded_array = np.pad(orig_array, 1, mode='constant', constant_values=np.nan)

    # create neigbhour arrays by shifting the original frame by the 8 different motion steps, and pseudo stacking
    # them into an empty 3d array
    neighbours = np.empty((8, height, width), dtype=np.float32)
    for i, (dx, dy) in enumerate(motion):
        neighbours[i] = padded_array[1 + dx : 1 + dx + height, 1 + dy : 1 + dy + width]
    
    comp_array = orig_array
    # compare the original array, remove nans from neigbour and original array, and keep array elements that are
    # within the threshold value
    compare = (~np.isnan(neighbours)) & (~np.isnan(comp_array)) & (np.abs(neighbours - comp_array) <= threshold)
    
    # extact array indices that meet the previous filtering criteria
    z_idx, x_idx, y_idx = np.where(compare)
    # see if needed might not be
    # if z_idx.size == 0:
    #     return np.empty((0, 4), dtype=np.int32)
    
    # get offsets to find nearest neighbours
    dx = motion[z_idx, 0]
    dy = motion[z_idx, 1]
    
    # get original coordinates and nearest neigbours that meet the above criteria
    ox = x_idx.astype(np.int32)
    oy = y_idx.astype(np.int32)
    nx = (ox + dx).astype(np.int32)
    ny = (oy + dy).astype(np.int32)

    return np.column_stack((ox, oy, nx, ny)).astype(np.int32)

def connected_components_from_edges(edges):
    
    if edges.size == 0:
        return []
    
    # turn all edge coordinates into nodes (root of set below)
    nodes = np.vstack([edges[:, :2], edges[:, 2:4]])
    # get the unique coordintes and their indices in the node array
    nodes_unique, inv = np.unique(nodes, axis=0, return_inverse=True)
    # extract the x and y coordinate
    x = inv[:edges.shape[0]]
    y = inv[edges.shape[0]:]

    # disjoint set (union find)
    # set parent/root nodes by number of unique coordinates
    parent = np.arange(nodes_unique.shape[0], dtype=np.int32)
    # needed for optimization of nodes below
    rank = np.zeros_like(parent)
    
    # https://heycoach.in/blog/working-of-union-find
    # https://medium.com/@dhleee0123/union-find-algorithm-with-python-and-optimize-d0fd3431cb18
    # recursive function to find root node
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    # include exit criteria for finding elements in each root node
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
    
    # update parent node
    for a, b in zip(x, y):
        union(a, b)
    
    # get final number of root nodes
    roots = np.array([find(x) for x in range(parent.shape[0])], dtype=np.int32)
    # sorts roots so the same roots are beside one another
    order = np.argsort(roots)
    roots_sorted = roots[order]
    # set a boundary split where the root value changes
    splits = np.flatnonzero(np.diff(roots_sorted)) + 1
    # split the ordered root set at boundaries
    groups = np.split(order, splits)
    
    # goes back to the unique coordinate list to get the x,y coordinates for each component in a group
    return [nodes_unique[g] for g in groups]

def patoms(single_frame_array, threshold = threshold_val):
    
    arr = single_frame_array.astype(np.float32, copy=False)

    arr = arr.copy()
    arr[arr == 0] = np.nan
    
    # gets edges and connected components (patom elements)
    edges = all_nn_edges(arr, threshold=threshold)
    comps = connected_components_from_edges(edges)

    # remove single pixels that have no nearest neighbours within threshold value
    comps = [c for c in comps if c.shape[0] >= 2]

    height, width = arr.shape
    # get array center values for determining segment
    center_x = (height - 1) / 2.0
    center_y = (width - 1) / 2.0
    num_segments = 16
    segment_width = 360.0 / num_segments
    
    out = []
    rng = np.random.default_rng()

    for coords in comps:
        i_vals = coords[:, 0].astype(np.float32)
        j_vals = coords[:, 1].astype(np.float32)
        # use the coordinates returned in the patom coordinates to get the original colour
        colours = arr[coords[:, 0], coords[:, 1]].astype(np.float32)

        # not sure if floor or ceiling is the better option - test if time
        x_mean = float(np.floor(i_vals.mean()))
        y_mean = float(np.floor(j_vals.mean()))
        
        # get min/max spread for x and y cooridnates in patom
        min_x = float(i_vals.min()); max_x = float(i_vals.max())
        min_y = float(j_vals.min()); max_y = float(j_vals.max())

        # normalisation from -1 to +1
        r_x = max(max_x - x_mean, x_mean - min_x)
        r_y = max(max_y - y_mean, y_mean - min_y)
        # had to set a check here as was getting nonsense back when a small patom has zero radius in x or y
        norm_x = (i_vals - x_mean) / r_x if r_x != 0 else np.full_like(i_vals, 0, dtype=np.float32)
        norm_y = (j_vals - y_mean) / r_y if r_y != 0 else np.full_like(j_vals, 0, dtype=np.float32)

        # calculate segment in unit circle, patom centroid falls in to
        angle_deg = (np.degrees(np.arctan2(center_y - y_mean, x_mean - center_x)) + 360.0) % 360.0
        angle_clockwise_from_north = (90.0 - angle_deg) % 360.0
        segment = angle_clockwise_from_north // segment_width
        
        # patom id
        patom_id = rng.random(dtype=np.float32)

        first_row  = np.array([patom_id, x_mean, y_mean, segment], dtype=np.float32)
        second_row = np.array([min_x, max_x, min_y, max_y], dtype=np.float32)

        patom_vals = np.column_stack((
            norm_x.astype(np.float32),
            norm_y.astype(np.float32),
            colours.astype(np.float32)
        ))
        patom_vals = np.hstack([patom_vals, np.full((patom_vals.shape[0], 1), np.nan, dtype=np.float32)])

        patom = np.vstack((first_row, second_row, patom_vals)).astype(np.float32)
        out.append(patom)

    return out

###########################################################################
from operator import itemgetter
from time import perf_counter, process_time, clock_gettime_ns, CLOCK_REALTIME
import random
import string

# choose image shape (must be large enough to include all x,y)
shape = (64, 64)

# number of sequences to import
n = 1
# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:n, ...]
print('loaded')
patoms_to_save = []
#generate patoms from sequences and save to disk

sequence = dataset[0]
original_frame = sequence[0]; flat = np.unique(original_frame.flatten()).tolist(); print('orig frame distinct vals',len(flat))
frame = sequence[0]
out_patoms = patoms(frame); print('num_patoms',len(out_patoms))

pixel_values = [i[:,3] for i in out_patoms]
print(len(pixel_values[0]))

unnorm_list = []
for i in out_patoms:
    #print(i[2:4,:])
    unnorm = unnormalise_xy(i)
    #print('norm', i[2:4,[4,5]], 'unnorm', unnorm[2:4,:])
    error = np.mean(i[2:,:2] != unnorm[2:,:2])
    #print('val error', error)
    diff_mask = i[2:,:2] != unnorm[2:,:2]
    diff_indices = np.argwhere(diff_mask)
    for idx in diff_indices:
        val_a = i[2:,:2][tuple(idx)]
        val_b = unnorm[2:,:2][tuple(idx)]
        diff = val_a - val_b
        print(f"Index {tuple(idx)}: A={val_a}, B={val_b}, Diff={diff}")
    unnorm_list.append(unnorm)

# check for duplicates in the output of the list of patoms
patoms_stacked = np.vstack([i[2:,:3] for i in out_patoms])
print('stacked patoms before de-dup', patoms_stacked.shape) # re collected patoms do not add back up to original frame size, why?
# remove duplicates
patoms_stacked = np.unique(patoms_stacked, axis=0)
print('stacked patoms after de-dup',patoms_stacked.shape) # re collected patoms do not add back up to original frame size, why?

# get unique x/y values in patoms stacked
x_vals = patoms_stacked[:,0]; print('pat stack unq x vals', np.unique(x_vals).tolist())

# check for duplicates in the output of the list of un-normalised patoms
unnorm_stacked = np.vstack([i[2:,:3] for i in unnorm_list])
print('stacked unnorm before de-dup', unnorm_stacked.shape) # re collected patoms do not add back up to original frame size, why?
# remove duplicates
unnorm_stacked = np.unique(unnorm_stacked, axis=0)
print('stacked unnorm after de-dup',unnorm_stacked.shape) # re collected patoms do not add back up to original frame size, why?


# recombine stacked patoms to check against original frame
h, w = shape

y = unnorm_stacked[:, 0].astype(np.int64, copy=False);y_list = np.unique(y).tolist(); print('patom stacked distinct vals', (y_list))
x = unnorm_stacked[:, 1].astype(np.int64, copy=False);x_list = np.unique(x).tolist(); print('patom stacked distinct vals', (x_list))
v = unnorm_stacked[:, 2]; print(v.shape);val_list = np.unique(v).tolist(); print('patom stacked distinct vals', (val_list))

fill_value = np.mean(original_frame)

out = np.full(shape, 0, dtype=int)
out[y, x] = v  # last occurrence wins due to NumPy assignment semantics
print((out.shape))

unique_out = np.unique(out).tolist()
print(unique_out)

import numpy as np
import matplotlib.pyplot as plt

img_patom = out
img_orig = original_frame

images = [original_frame, out]

cmap = 'gray'

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

im = None
for ax, img in zip(axes, images):
    im = ax.imshow(img, cmap=cmap)
    ax.axis('off')

fig.suptitle('Orig Frame vs Recreated Frame', fontsize=15)
plt.show()
plt.imsave('patom_test.png', img, cmap='gray')


# print(out[:5,:5])
# print(original_frame[:5,:5])
orig_frame_patoms_match = np.mean(out != original_frame)
print('% mismatch with zeros',orig_frame_patoms_match)
# remove zeros and calculate mistmatch
out_nonzero = out.copy().astype('float32')
out_nonzero[out_nonzero == 0] = float(np.nan)
orig_nonzero = original_frame.copy().astype('float32')
orig_nonzero[orig_nonzero == 0] = float(np.nan)

mask = ~np.isnan(out_nonzero) & ~np.isnan(orig_nonzero)
out_nonzero = out_nonzero[mask]
orig_nonzero = orig_nonzero[mask]

orig_frame_patoms_match_nonzero = np.mean(out_nonzero != orig_nonzero)
print('% mismatch without zeros',orig_frame_patoms_match_nonzero)
diff_mask = out != original_frame
diff_indices = np.argwhere(diff_mask)

print('orig nonzero', orig_nonzero.shape)
print('out nonzero', out_nonzero.shape)
diffs = []

for idx in diff_indices:
    val_a = out[tuple(idx)]
    val_b = original_frame[tuple(idx)]
    diff = val_a - val_b
    #print(f"Index {tuple(idx)}: A={val_a}, B={val_b}, Diff={diff}")
    diffs.append(diff)

print('% mismatch nonzero:', 1 - ((orig_nonzero.shape[-1] - len(diffs)) / orig_nonzero.shape[-1]) )