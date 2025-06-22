import numpy as np
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME

threshold = 0.0005 #0.00005 -- need to reasses this when we get live data
motion = [[-1, -1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]

def snapshot(x_len, y_len, single_frame_array, i):
    orig_array = single_frame_array
    
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

    orig_array = orig_array[1:-1, 1:-1]
    arr = orig_array[ia:ib, ja:jb]
    comp =  orig_array[xa:xb, ya:yb]
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
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i/y_len, orig_loc_j/x_len))# 7.5 MiB
    orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
    loc2 = list(zip(true_indices[0]/y_len, true_indices[1]/x_len)) # 7.5 MiB
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1]]) # 2.5 MiB
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))# 3.1 MiB
    out = orig_vals_inds + tnn_vals_inds
    
    return out

def patoms2d(x_len, y_len, single_frame_array, frame_ind):
    items = [(x_len, y_len, single_frame_array, i) for i in range(8)]
    # with multiprocessing
    atime = perf_counter()
    with Pool(processes=cpu_count()) as pool:
        res = pool.starmap(snapshot, items)

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
        if len(s) >= 10: # add filter for patoms of less than 10 pixels
            s_res.append(s)

    # then need to obtain a normalised distance for all points from the 'center' of the pattern
    norm_patoms = []
    for patom_ind, pat in enumerate(s_res):
        pat_len = len(pat)
        x_vals = [p[1][0] for p in pat]; min_x = min(x_vals); max_x = max(x_vals)
        norm_x = [2 * (x - min_x) / (max_x - min_x) - 1 for x in x_vals]
        y_vals = [p[1][1] for p in pat]; min_y = min(y_vals); max_y = max(y_vals)
        norm_y = [2 * (x - min_y) / (max_y - min_y) - 1 for x in y_vals]

        pattern_centroid_x = sum(norm_x)/pat_len
        pattern_centroid_y = sum(norm_y)/pat_len
        
        patom_time = clock_gettime_ns(CLOCK_REALTIME)
        patom = [norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, patom_ind, frame_ind, patom_time]
        norm_patoms.append(patom)

    print("Time to get 2D patterns with multiprocessing (secs):", (perf_counter()-atime))

    return norm_patoms