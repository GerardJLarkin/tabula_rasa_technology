## add ignore warnings for now, will remove and debug once full algorithm is complete
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import numpy as np
from multiprocessing import Pool, cpu_count
import sys
import sqlite3

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

## create in memory 2d database
con2d = sqlite3.connect("ref2d_v4.db")
cur2d = con2d.cursor()

fps30 = [1] * 360

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

def ref_patoms2d(x_len, y_len, single_frame_array, frame_ind):
    items = [(x_len, y_len, single_frame_array, i) for i in range(8)]
    # with multiprocessing
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
        norm_x = np.array([2 * (x - min_x) / (max_x - min_x) - 1 for x in x_vals]).reshape(pat_len,1)
        y_vals = [p[1][1] for p in pat]; min_y = min(y_vals); max_y = max(y_vals)
        norm_y = np.array([2 * (x - min_y) / (max_y - min_y) - 1 for x in y_vals]).reshape(pat_len,1)
        pattern_centroid_x = np.array([sum(norm_x)/pat_len] * pat_len).reshape(pat_len,1)
        pattern_centroid_y = np.array([sum(norm_y)/pat_len] * pat_len).reshape(pat_len,1)
        patom_ind = np.array([patom_ind] * pat_len).reshape(pat_len,1)
        frame_ind_arr = np.array([frame_ind] * pat_len).reshape(pat_len,1)
        
        patom_time = np.array([clock_gettime_ns(CLOCK_REALTIME)] * pat_len).reshape(pat_len,1)

        cond_x = [norm_x >= 0, norm_x < 0]
        choice_x = [1, 2]
        quadx = np.select(cond_x, choice_x)
        cond_y = [norm_y >= 0, norm_y < 0]
        choice_y = [3, 4]
        quady = np.select(cond_y, choice_y)
        quad = np.hstack([quadx, quady])
        quad = np.array([int(f"{a}{b}") for a, b in quad]).reshape(pat_len,1)

        # Get unique values and their counts
        unique_values, counts = np.unique(quad, return_counts=True)
        # Create a dictionary to map values to their counts
        value_to_count = {value: count for value, count in zip(unique_values, counts)}
        # Add a new column with the counts
        quad_cnt = np.array([value_to_count[value] for value in quad.flatten()]).reshape(pat_len, 1)
        
        # 9 columns (9,10,11,12,13,14,15,16,17)
        patom_array = np.hstack([norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, patom_ind, frame_ind_arr, patom_time]).astype(np.float32)
        
        norm_patoms.append(patom_array)

    return norm_patoms

table_num_seq = 0
for i in range(500):
    rand_array = np.random.random((1, 720, 1280))
    y_len = rand_array.shape[1]
    x_len = rand_array.shape[2]
    frame_patoms = ref_patoms2d(x_len, y_len, rand_array[0,:,:], i)
    num_patoms = len(frame_patoms)
    for j in range(num_patoms):
        table_num = str(table_num_seq).zfill(6)
        patom = frame_patoms[j].tolist()
        # patom_to_table = patom_to_table_func(patom)
        cur2d.execute(f"CREATE TABLE pat_2d_ref_{table_num}(x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, patom_ind, frame_ind, patom_time);")
        cur2d.executemany(f"INSERT INTO pat_2d_ref_{table_num}(x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, patom_ind, frame_ind, patom_time) VALUES (?,?,?,?,?,?,?,?,?)", patom)
        cur2d.execute(f"CREATE TABLE pat_2d_{table_num}(x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, patom_ind, frame_ind, patom_time, xc_d, yc_d, x_d, y_d);")
        table_num_seq += 1 

con2d.commit() 
   

# print(max([table for (table,) in cur2d.execute("select name from sqlite_master where type='table';").fetchall()]))
# print([x[0] for x in cur2d.execute("select * from pat_2d_ref_000011;").fetchall()])
con2d.close()