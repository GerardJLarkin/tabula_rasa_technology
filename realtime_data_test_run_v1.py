###### realtime data test run
## add ignore warnings for now, will remove and debug once full algorithm is complete
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter, process_time, clock_gettime_ns, CLOCK_REALTIME
# from operator import itemgetter
import numpy as np
# import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import sys
import sqlite3
# import cv2 as cv

from cProfile import Profile
from pstats import SortKey, Stats

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')


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

def patoms2d(x_len, y_len, single_frame_array, frame_ind):
    items = [(x_len, y_len, single_frame_array, i) for i in range(8)]
    # with multiprocessing
    atime = perf_counter(), process_time()
    with Pool(processes=4) as pool:
        res = pool.starmap(snapshot, items)

    # combine the outputs of each nearest neighbour function
    combined_output = np.vstack((res))
    combined_output = np.unique(combined_output, axis=0)
    combined_output = combined_output[combined_output[:,0].argsort()] 
    # norm_x = 2 * ((combined_output[:,1] - combined_output[:,1].min()) / (combined_output[:,1].max() - combined_output[:,1].min())) - 1 
    # norm_y = 2 * ((combined_output[:,2] - combined_output[:,2].min()) / (combined_output[:,2].max() - combined_output[:,2].min())) - 1 
    # combined_output = np.column_stack((combined_output[:,0], norm_x, norm_y))
    
    # split patoms based on colour threshold
    differences = np.diff(combined_output[:, 0])
    split_indices = np.where(differences > threshold)[0] + 1
    chunks = np.split(combined_output, split_indices)
    
    norm_patoms = []
    for i in chunks:
        x_vals = i[:,1]; y_vals = i[:,2]
        pat_len = i.shape[0]

        norm_x = 2 * ((x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())) - 1 
        norm_y = 2 * ((y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())) - 1 
        
        pattern_centroid_x = np.array([x_vals.mean()] * pat_len).reshape(pat_len,1)
        pattern_centroid_y = np.array([y_vals.mean()] * pat_len).reshape(pat_len,1)
        frame_ind_arr = np.array([frame_ind] * pat_len).reshape(pat_len,1)
        colours = i[:,0]

        cond_x = [norm_x >= 0, norm_x < 0]
        choice_x = [1, 2]
        quadx = np.select(cond_x, choice_x)
        #print(quadx)
        cond_y = [norm_y >= 0, norm_y < 0]
        choice_y = [3, 4]
        quady = np.select(cond_y, choice_y)
        #print(quady)
        quad = np.column_stack((quadx, quady))
        #print(quad)
        quad = np.array([int(f"{a}{b}") for a, b in quad]).reshape(pat_len,1)

        # Get unique values and their counts
        unique_values, counts = np.unique(quad, return_counts=True)
        # Create a dictionary to map values to their counts
        value_to_count = {value: count for value, count in zip(unique_values, counts)}
        # Add a new column with the counts
        quad_cnt = np.array([value_to_count[value] for value in quad.flatten()]).reshape(pat_len, 1)
        # 9 columns (0,1,2,3,4,5,6,7)
        patom_array = np.column_stack((colours, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr)).astype('float32')
        ind = np.lexsort((patom_array[:,1], patom_array[:,2]))
        patom_array = patom_array[ind]
        norm_patoms.append(patom_array)

    #stacked_patoms = np.vstack(norm_patoms)

    print("Real Time to get 2D patterns with multiprocessing (secs):", (perf_counter()-atime[0]))
    print("CPU Time to get 2D patterns with multiprocessing (secs):", (process_time()-atime[1]))

    return norm_patoms


def pattern_compare_2d(new_patom, ref_array):
    # new patom: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr]
    # ref patoms: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr]
    # Extract the category column
    categories1 = new_patom[:, 5]
    categories2 = ref_array[:, 5]
    # Perform a Cartesian join
    # Expand dimensions to compare categories using broadcasting
    category_match = categories1[:, None] == categories2[None, :]
    # Get indices where categories match
    indices1, indices2 = np.where(category_match)
    # Select matching rows
    matched_array1 = new_patom[indices1]
    matched_array2 = ref_array[indices2]
    # Concatenate horizontally to form the final joined result
    cart = np.hstack((matched_array1, matched_array2))

    cart_x_diff = np.absolute(cart[:,1] - cart[:,9])
    cart_y_diff = np.absolute(cart[:,2] - cart[:,10])
    cart_quad_len = cart[:,6] * cart[:,14]
    cart_xc_d = np.absolute(cart[:,3] - cart[:,11])
    cart_yc_d = np.absolute(cart[:,4] - cart[:,12])
    cart_x_diff_sum = np.resize(cart_x_diff.sum(), cart_quad_len.shape[0])
    cart_y_diff_sum = np.resize(cart_y_diff.sum(), cart_quad_len.shape[0])
    cart_x_d = cart_x_diff_sum / cart_quad_len
    cart_y_d = cart_y_diff_sum / cart_quad_len
    cond = [(cart_xc_d <= 0.2) & (cart_yc_d <= 0.2) & (cart_x_d <= 0.3) & (cart_y_d <= 0.3)]
    choice = [1]
    cart_similar = np.select(cond, choice, 0)
    cart_arr = np.unique(np.column_stack((cart[:,:8], cart_xc_d, cart_yc_d, cart_x_d, cart_y_d, cart[:,15], cart_similar)), axis=0)
    
    return cart_arr


## access the camera to get video stream
# cap = cv.VideoCapture(0)

# connect to 2d database
con2d = sqlite3.connect("ref2d_v5.db")
cur2d = con2d.cursor()

working_ref_patoms = []
con2dref = sqlite3.connect("ref2d_v5.db")
cur2dref = con2dref.cursor()
ref_names = [name for (name,) in cur2dref.execute("select name from sqlite_master where type='table' and name like '%ref%';").fetchall()][:40]
for i in ref_names:
    table = cur2dref.execute(f"select * from {i};").fetchall()
    table_array = np.array(table).astype(np.float32)
    working_ref_patoms.append(table_array)

ref_patoms_array = np.vstack(working_ref_patoms).astype('float32')
ref_indices = np.unique(ref_patoms_array[:,7],axis=0)

with Profile() as profile:
    s = perf_counter(), process_time()
    val = 0
    while val < 3:
        frame = np.random.randint(0, 256, (720, 1280), dtype=np.uint8) # 307200 bytes
        # print(frame.nbytes)
        # ret, frame = cap.read()
        # frame = (frame[..., 0] << 16) | (frame[..., 1] << 8) | frame[..., 2]
        # print(frame.nbytes)
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        x_len = frame.shape[0]
        y_len = frame.shape[1]
        ####################### FIRST TASK: FIND PATTERNS IN FRAME ######################
        frame_patoms = patoms2d(x_len, y_len, frame, val)
        # output: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr]
        #frame_patoms_arr = np.vstack((frame_patoms))
        num_patoms = len(frame_patoms)
        ############## SECOND TASK: COMPARE NEW PATOMS AGAINST REF PATOMS ###############
        atime = perf_counter(), process_time()
        with Pool(processes=8) as pool:
            items = [(frame_patoms[i], ref_patoms_array) for i in range(num_patoms)]
            comp_results = pool.starmap(pattern_compare_2d, items) #e.g. ['pcol','px','py','pxc','pyc','pq','pqlen','pfind','xc_d','yc_d','x_d','y_d','rfind','similar']
            print("Real Time to compare 2D patterns with multiprocessing (secs):", (perf_counter()-atime[0]))
            print("CPU Time to compare 2D patterns with multiprocessing (secs):", (process_time()-atime[1]))
            # loop through the output of the comparison function
            witime = perf_counter(), process_time()
            for ix, i in enumerate(comp_results):
                table_data = np.unique(i[:,:-2], axis=0)
                existing_tables = [names for (names,) in cur2d.execute("select name from sqlite_master where type='table' and name not like '%ref%';").fetchall()]
                next_table_num = int(existing_tables.pop(-1)[-6:]) + 1 
                next_table_num = str(next_table_num).zfill(6)
                ref_patom_ind = i[:,12].astype('int32')
                ref_patom_ind = np.unique(ref_patom_ind).tolist() # contains list of all ref patoms used in comparison
                #extract reference tables that are similar to the patom in the current iteration
                match_patoms = i[i[:,13] == 1]
                if match_patoms.shape[0] > 0:
                    for k in ref_patom_ind:
                        table_num = ref_names[k][-6:]
                        cur2d.executemany(f"INSERT INTO pat_2d_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, frame_ind, \
                                            xc_d, yc_d, x_d, y_d) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", table_data)
                # if no matching ref patoms, write new patom to new table
                else:
                    cur2d.execute(f"CREATE TABLE pat_2d_{next_table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, frame_ind, \
                                xc_d, yc_d, x_d, y_d);")
                    cur2d.executemany(f"INSERT INTO pat_2d_{next_table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, frame_ind, \
                                    xc_d, yc_d, x_d, y_d) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", table_data)    
                
                
            ## major bottleneck is the insert/create steps
            print("Real Time to write/insert patoms to existing/new tables in database (secs):", (perf_counter()-witime[0]))
            print("CPU Time to write/insert patoms to existing/new tables in database (secs):", (process_time()-witime[1]))
        val += 1
    
    con2d.commit()
    print("Real Time to process 1 second of data against 40 reference patoms(secs):", (perf_counter()-s[0]))
    print("CPU Time to process 1 second of data against 40 reference patoms(secs):", (process_time()-s[1]))
    #print("Time to get patoms from 1 seconds of data (mins):", (perf_counter()-s)/60)
    con2d.close()
    con2dref.close()

    (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.CUMULATIVE)
        .print_stats()
    )