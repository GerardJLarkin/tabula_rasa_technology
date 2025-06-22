# https://stackoverflow.com/questions/70172677/compare-value-in-a-2d-orig_array-to-nearby-values
import numpy as np
from operator import itemgetter
import math
from time import perf_counter

from multiprocessing import Pool, TimeoutError
import time
import os

np.random.seed(5555)
rand_array = np.random.random((30, 720, 1280))
# print(rand_array.nbytes)

# time_array = np.load('/home/gerard/Desktop/capstone_project/norm_file.npy')
# print(time_array.nbytes)

motion = [[-1, -1, -1], [0, -1, -1], [1, -1, -1], [-1, 0, -1], [0, 0, -1], [1, 0, -1], [-1, 1, -1], 
          [0, 1, -1], [1, 1, -1], [-1, -1, 0], [0, -1, 0], [1, -1, 0], [-1, 0, 0], [1, 0, 0], [-1, 1, 0], 
          [0, 1, 0], [1, 1, 0], [-1, -1, 1], [0, -1, 1], [1, -1, 1], [-1, 0, 1], [0, 0, 1], [1, 0, 1], 
          [-1, 1, 1], [0, 1, 1], [1, 1, 1]]

def snapshot_pattern(array):
    strt_stack = perf_counter()
    threshold = 0.00005
    orig_array = array
    
    outs = []
    for i in range(26):
        indxs = motion[i]
        if indxs[0] == 1:
            ia = None; ib = -1
            xa = 1; xb = None
        if indxs[0] == -1:
            ia = 1; ib = None
            xa = None; xb = -1      
        if indxs[1] == 1:
            ja = None; jb = -1
            ya = 1; yb = None
        if indxs[1] == -1:
            ja = 1; jb = None
            ya = None; yb = -1
        if indxs[2] == 1:
            ka = None; kb = -1
            za = 1; zb = None
        if indxs[2] == -1:
            ka = 1; kb = None
            za = None; zb = -1  

        arr = orig_array[ia:ib, ja:jb, ka:kb]
        comp =  orig_array[xa:xb, ya:yb, za:zb]
        truth = abs(comp - arr) <= threshold # heavy not too bad ~27MiB
        true_indices = np.asarray(truth).nonzero()
        def get_orig_loc_i(x):
            if indxs[0] == 1:
                return x+1
            if indxs[0] == 0:
                return x
            if indxs[0] == -1:
                return x-1
        def get_orig_loc_j(x):
            if indxs[1] == 1:
                return x+1
            if indxs[1] == 0:
                return x
            if indxs[1] == -1:
                return x-1
        def get_orig_loc_k(x):
            if indxs[2] == 1:
                return x+1
            if indxs[2] == 0:
                return x
            if indxs[2] == -1:
                return x-1
        orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
        orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
        orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
        get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
        loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))# 7.5 MiB
        orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
        loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2])) # 7.5 MiB
        get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
        tnn_vals_inds = list(zip(get_tnn_vals, loc2))# 3.1 MiB
        out = orig_vals_inds + tnn_vals_inds
        outs.append(out)
    
    # combine the outputs of each nearest neighbour function
    combined_output = sorted(set([i for x in outs for i in x]))
    
    # split list when value between subsequent elements is greater than threshold
    res, last = [[]], None
    for x in combined_output:
        if last is None or abs(last - x[0]) <= threshold: #runtime warning here
            res[-1].append(x)
        else:
            res.append([x])
        last = x[0]

    # sort the lists of tuples based on the indices (need to get indices as tuple)
    s_res = []
    for i in res:
        s = sorted(i, key=itemgetter(1))
        if len(s) >= 10:
            s_res.append(s)

    # then need to obtain a normalised distance for all points from the 'center' of the pattern
    norm_patoms = []
    for pat in s_res:
        x = [p[1][0] for p in pat]
        min_x = min(x); max_x = max(x)
        norm_x = [(i - min_x)/(max_x - min_x) for i in x]
        y = [p[1][1] for p in pat]
        min_y = min(y); max_y = max(y)
        norm_y = [(i - min_y)/(max_y - min_y) for i in y]
        z = [p[1][2] for p in pat]
        min_z = min(z); max_z = max(z)
        norm_z = [(i - min_z)/(max_z - min_z) for i in z]
        centroid = list((sum(x) / len(pat), sum(y) / len(pat), sum(z) / len(pat)))
        centroid_norm = list((sum(norm_x) / len(pat), sum(norm_y) / len(pat), sum(norm_z) / len(pat)))
        centroid_list = [tuple([0.0]+centroid+centroid_norm+[0.0])]
        loc = [p[1] for p in pat]
        dist = list(map(lambda x: math.dist(centroid, list(x)), loc))
        tot = sum(dist)
        # normalised distance value is from centroid
        norm_dist = [i/tot for i in dist]
        val = [p[0] for p in pat]
        patom = list(zip(val, x, y, z, norm_x, norm_y, norm_z, norm_dist))
        patom = patom + centroid_list
        norm_patoms.append(patom)

    end_stack = perf_counter()

    stacked_time = end_stack - strt_stack
    print('Took this many seconds: ', stacked_time)

    return norm_patoms

# if __name__ == '__main__':
#     # start 8 worker processes
#     with Pool(processes=8) as pool:
#         # apply pattern recognition across all cores
#         output = pool.map(snapshot_pattern(rand_array), range(26))

# # output = snapshot_pattern(rand_array)
# print(len(output))
