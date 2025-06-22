import numpy as np
from multiprocessing import Pool, cpu_count
import math
from operator import itemgetter
from time import perf_counter
import sys
import numpy as np
from time import perf_counter
import csv
import os

sys.path.append('/home/gerard/Desktop/capstone_project')

# for directory in range(10):
#     path = f'/home/gerard/Desktop/capstone_project/patoms_{directory}'
#     if not os.path.exists(path):
#         os.mkdir(path)
np.random.seed(5555)
rand_array = np.random.random((30, 720, 1280))
z_len = rand_array.shape[0]
y_len = rand_array.shape[1]
x_len = rand_array.shape[2]

threshold = 0.00005 #0.00005
motion = [[-1, -1, -1], [0, -1, -1], [1, -1, -1], [-1, 0, -1], [0, 0, -1], [1, 0, -1], [-1, 1, -1], 
        [0, 1, -1], [1, 1, -1], [-1, -1, 0], [0, -1, 0], [1, -1, 0], [-1, 0, 0], [1, 0, 0], [-1, 1, 0], 
        [0, 1, 0], [1, 1, 0], [-1, -1, 1], [0, -1, 1], [1, -1, 1], [-1, 0, 1], [0, 0, 1], [1, 0, 1], 
        [-1, 1, 1], [0, 1, 1], [1, 1, 1]]

def snapshot_pattern(i):
    orig_array = rand_array
    
    indxs = motion[i]
    if indxs[0] == 1:
        ia = None; ib = -1; xa = 1; xb = None
    if indxs[0] == 0:
        ia = None; ib = None; xa = None; xb = None
    if indxs[0] == -1:
        ia = 1; ib = None; xa = None; xb = -1      
    if indxs[1] == 1:
        ja = None; jb = -1; ya = 1; yb = None
    if indxs[1] == 0:
        ja = None; jb = None; ya = None; yb = None
    if indxs[1] == -1:
        ja = 1; jb = None; ya = None; yb = -1
    if indxs[2] == 1:
        ka = None; kb = -1; za = 1; zb = None
    if indxs[2] == 0:
        ka = None; kb = None; za = None; zb = None
    if indxs[2] == -1:
        ka = 1; kb = None; za = None; zb = -1  

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
    loc1 = list(zip(orig_loc_i/z_len, orig_loc_j/y_len, orig_loc_k/x_len))# 7.5 MiB
    orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
    loc2 = list(zip(true_indices[0]/z_len, true_indices[1]/y_len, true_indices[2]/x_len)) # 7.5 MiB
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))# 3.1 MiB
    out = orig_vals_inds + tnn_vals_inds

    return out


def pre_patom(pat):
    pat_len = len(pat)
    x = [p[1][0] for p in pat]
    mean_x = sum(x)/pat_len; min_x = min(x); max_x = max(x)
    norm_x = [(i - min_x)/(max_x - min_x) for i in x]
    y = [p[1][1] for p in pat]
    mean_y = sum(y)/pat_len; min_y = min(y); max_y = max(y)
    norm_y = [(i - min_y)/(max_y - min_y) for i in y]
    z = [p[1][2] for p in pat]
    mean_z = sum(z)/pat_len; min_z = min(z); max_z = max(z)
    norm_z = [(i - min_z)/(max_z - min_z) for i in z]
    position_centroid = list((mean_x, mean_y, mean_z))
    pattern_centroid = list((sum(norm_x)/pat_len, sum(norm_y)/pat_len, sum(norm_z)/pat_len))
    centroid_list = [tuple([0.0]+position_centroid+pattern_centroid+[0.0])]
    loc_norm = list(zip(norm_x, norm_y, norm_z))
    norm_dist = list(map(lambda x: math.dist(pattern_centroid, list(x)), loc_norm))
    val = [p[0] for p in pat]
    patom = list(zip(val, x, y, z, norm_x, norm_y, norm_z, norm_dist))
    patom = patom + centroid_list

    return patom

def patoms():
    # with multiprocessing
    atime = perf_counter()
    with Pool(processes=cpu_count()) as pool:
        res = pool.map(snapshot_pattern, range(26))

    # combine the outputs of each nearest neighbour function
    combined_output = sorted(set([i for x in res for i in x]))

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
        pat_len = len(pat)
        x = [p[1][0] for p in pat]
        mean_x = sum(x)/pat_len; min_x = min(x); max_x = max(x)
        norm_x = [(i - min_x)/(max_x - min_x) for i in x]
        y = [p[1][1] for p in pat]
        mean_y = sum(y)/pat_len; min_y = min(y); max_y = max(y)
        norm_y = [(i - min_y)/(max_y - min_y) for i in y]
        z = [p[1][2] for p in pat]
        mean_z = sum(z)/pat_len; min_z = min(z); max_z = max(z)
        norm_z = [(i - min_z)/(max_z - min_z) for i in z]
        position_centroid = list((mean_x, mean_y, mean_z))
        pattern_centroid = list((sum(norm_x)/pat_len, sum(norm_y)/pat_len, sum(norm_z)/pat_len))
        centroid_list = [tuple([0.0]+position_centroid+pattern_centroid+[0.0])]
        loc_norm = list(zip(norm_x, norm_y, norm_z))
        norm_dist = list(map(lambda x: math.dist(pattern_centroid, list(x)), loc_norm))
        val = [p[0] for p in pat]
        patom = list(zip(val, x, y, z, norm_x, norm_y, norm_z, norm_dist))
        patom = patom + centroid_list
        norm_patoms.append(patom)
    
    # with Pool(processes=cpu_count()) as pool:
    #     norm_patoms = pool.map(pre_patom, s_res)

    # structure each file with the pixel value, inde, etc. as a single row
    # for ind, pat in enumerate(norm_patoms):
    #     with open(f'/home/gerard/Desktop/capstone_project/patoms/pat{ind}.csv', 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         writer.writerows(pat)

    print("Time to get patterns with multiprocessing (mins):", (perf_counter()-atime)/60)

    return norm_patoms

#patoms()