## add ignore warnings for now, will remove and debug once full algorithm is complete
# import warnings
# warnings.filterwarnings("ignore")

## import packages/libraries
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import numpy as np
from multiprocessing import Pool, cpu_count
import sys
import sqlite3
import cv2 as cv
import random

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

## create in memory 2d database
con2d = sqlite3.connect("ref2d_v5.db")
cur2d = con2d.cursor()

threshold = 0.00005 #0.00005 -- need to reasses this when we get live data
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
    
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]) # column 2
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1]) # column 3
    orig_vals = orig_array[orig_loc_i, orig_loc_j] # column 1
    # orig_loc_i = orig_loc_i / y_len
    # orig_loc_j = orig_loc_j / x_len

    originals = np.column_stack((orig_vals, orig_loc_i, orig_loc_j))
    
    tnn_loc_i = true_indices[0] #/ y_len
    tnn_loc_j = true_indices[1] #/ x_len
    tnn_vals = orig_array[true_indices[0], true_indices[1]]
    
    nearest_neigbours = np.column_stack((tnn_vals, tnn_loc_i, tnn_loc_j))

    orig_nn = np.vstack((originals, nearest_neigbours))
    
    return orig_nn

def ref_patoms2d(x_len, y_len, single_frame_array, frame_ind):
    items = [(x_len, y_len, single_frame_array, i) for i in range(8)]
    # with multiprocessing
    atime = perf_counter()
    with Pool(processes=4) as pool:
        res = pool.starmap(snapshot, items)

    # combine the outputs of each nearest neighbour function
    combined_output = np.vstack((res))
    combined_output = np.unique(combined_output, axis=0)
    combined_output = combined_output[combined_output[:,0].argsort()]
    
    # split patoms based on colour threshold
    differences = np.diff(combined_output[:, 0],axis=0)
    split_indices = np.where(differences > threshold)[0] + 1
    
    chunks = np.split(combined_output, split_indices)
    
    norm_patoms = []
    for ix, i in enumerate(chunks):
        x_vals = i[:,1]; y_vals = i[:,2]
        pat_len = i.shape[0]
        
        norm_x = 2 * ((x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())) - 1 
        norm_y = 2 * ((y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())) - 1 
        
        pattern_centroid_x = np.array([norm_x.mean()] * pat_len).reshape(pat_len,1)
        pattern_centroid_y = np.array([norm_y.mean()] * pat_len).reshape(pat_len,1)
        sequence_ind = np.array([frame_ind] * pat_len).reshape(pat_len,1)
        colours = i[:,0]

        coords = np.column_stack((norm_x, norm_y))
        cond = [
                ((coords[:,0] < 1.0)          & (coords[:,0] >= 0.92387953)   & (coords[:,1] > 0.0)          & (coords[:,1] <= 0.38268343)), 
                ((coords[:,0] < 0.92387953)   & (coords[:,0] >= 0.70710678)   & (coords[:,1] > 0.38268343)   & (coords[:,1] <= 0.70710678)), 
                ((coords[:,0] < 0.70710678)   & (coords[:,0] >= 0.38268343)   & (coords[:,1] > 0.70710678)   & (coords[:,1] <= 0.92387953)),                 
                ((coords[:,0] < 0.38268343)   & (coords[:,0] >= 0.0)          & (coords[:,1] > 0.92387953)   & (coords[:,1] <= 1.0)),                 
                ((coords[:,0] < 0.0)          & (coords[:,0] >= -0.38268343)  & (coords[:,1] < 1.0)          & (coords[:,1] >= 0.92387953)),
                ((coords[:,0] < -0.38268343)  & (coords[:,0] >= -0.70710678)  & (coords[:,1] < 0.92387953)   & (coords[:,1] >= 0.70710678)),
                ((coords[:,0] < -0.70710678)  & (coords[:,0] >= -0.92387953)  & (coords[:,1] < 0.70710678)   & (coords[:,1] >= 0.38268343)),
                ((coords[:,0] < -0.92387953)  & (coords[:,0] >= -1.0)         & (coords[:,1] < 0.38268343)   & (coords[:,1] >= 0.0)), 
                ((coords[:,0] > -1.0)         & (coords[:,0] <= -0.92387953)  & (coords[:,1] < 0.0)          & (coords[:,1] >= -0.38268343)), 
                ((coords[:,0] > -0.92387953)  & (coords[:,0] <= -0.70710678)  & (coords[:,1] < -0.38268343)  & (coords[:,1] >= -0.70710678)), 
                ((coords[:,0] > -0.70710678)  & (coords[:,0] <= -0.38268343)  & (coords[:,1] < -0.70710678)  & (coords[:,1] >= -0.92387953)), 
                ((coords[:,0] > -0.38268343)  & (coords[:,0] <= 0.0)          & (coords[:,1] < -0.92387953)  & (coords[:,1] >= -1.0)), 
                ((coords[:,0] > 0.0)          & (coords[:,0] <= 0.38268343)   & (coords[:,1] > -1.0)         & (coords[:,1] <= -0.92387953)), 
                ((coords[:,0] > 0.38268343)   & (coords[:,0] <= 0.70710678)   & (coords[:,1] > -0.92387953)  & (coords[:,1] <= -0.70710678)), 
                ((coords[:,0] > 0.70710678)   & (coords[:,0] <= 0.92387953)   & (coords[:,1] > -0.70710678)  & (coords[:,1] <= -0.38268343)), 
                ((coords[:,0] > 0.92387953)   & (coords[:,0] <= 1.0)          & (coords[:,1] > -0.38268343)  & (coords[:,1] <= 0.0))               
               ]
        choice = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        segment = np.select(cond, choice)
        
        # Get unique values and their counts
        unique_values, counts = np.unique(segment, return_counts=True)
        # Create a dictionary to map values to their counts
        value_to_count = {value: count for value, count in zip(unique_values, counts)}
        # Add a new column with the counts
        segment_cnt = np.array([value_to_count[value] for value in segment.flatten()]).reshape(pat_len, 1)
        # 9 columns (0,1,2,3,4,5,6,7)
        patom_array = np.column_stack((colours, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, segment, segment_cnt, sequence_ind)).astype('float32')

        norm_patoms.append(patom_array)

    return norm_patoms

## access the camera to get video stream
# cap = cv.VideoCapture(0)
table_num_seq = 0
val = 0
while val <= 30:
    # ret, frame = cap.read()
    # frame = (frame[..., 0] << 16) | (frame[..., 1] << 8) | frame[..., 2]
    frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8) 
    frame = (frame - frame.min()) / (frame.max() - frame.min())
    x_len = frame.shape[0]
    y_len = frame.shape[1]
    patom_id = random.randint(0, 899)
    frame_patoms = ref_patoms2d(x_len, y_len, frame, patom_id)
    for patom in frame_patoms:
        diff_cols = np.resize(np.array([0.0,0.0,0.0,0.0,0.0]), (patom.shape[0],5))
        nonref_patom = np.hstack((patom, diff_cols))
        table_num = str(table_num_seq).zfill(6)
        cur2d.execute(f"CREATE TABLE ref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind);")
        cur2d.executemany(f"INSERT INTO ref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind) VALUES (?,?,?,?,?,?,?,?)", patom)
        cur2d.execute(f"CREATE TABLE nonref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind, col_d, xc_d, yc_d, x_d, y_d);")
        cur2d.executemany(f"INSERT INTO nonref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind, col_d, xc_d, yc_d, x_d, y_d) \
                          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", nonref_patom)
        table_num_seq += 1 

    val += 1
    print(val)

con2d.commit() 
# print(max([table for (table,) in cur2d.execute("select name from sqlite_master where type='table';").fetchall()]))
# print(np.array(cur2d.execute("select * from ref_000555;").fetchall()))

# table_names = [table for (table,) in cur2d.execute("select name from sqlite_master where type='table';").fetchall()]
# table_rows = []
# for i in table_names:
#     row_count = cur2d.execute(f"select count(*) from {i};").fetchall()[0][0]
#     table_rows.append(row_count)
# print(min(table_rows), max(table_rows))
con2d.close()