import numpy as np
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import perf_counter, process_time, clock_gettime_ns, CLOCK_REALTIME
import random
import string

threshold = 0.0005 #0.00005
motion = np.array([[-1, -1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]])

def snapshot(single_frame_array, i):

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
    truth = abs(comp - arr) <= threshold
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
    orig_vals = orig_array[orig_loc_i, orig_loc_j]

    originals = np.column_stack((orig_vals, orig_loc_i, orig_loc_j))
    
    tnn_loc_i = true_indices[0]
    tnn_loc_j = true_indices[1]
    tnn_vals = orig_array[true_indices[0], true_indices[1]]
    
    nearest_neigbours = np.column_stack((tnn_vals, tnn_loc_i, tnn_loc_j))

    orig_nn = np.vstack((originals, nearest_neigbours))
    
    return orig_nn

def patoms(single_frame_array):
    items = [(single_frame_array, i) for i in range(8)]
    # with multiprocessing
    with Pool(processes=2) as pool:
        res = pool.starmap(snapshot, items)

    # combine the outputs of each nearest neighbour function
    combined_output = np.vstack((res))
    combined_output = np.unique(combined_output, axis=0)
    combined_output = combined_output[combined_output[:,0].argsort()]

    ######################################################################
    ######################################################################
    ## adding in section to save combined oupput to file for inspection ##
    ## only saves last file (all others are overwritten)                ##
    np.savetxt(
    "inspection_data_combined_output.csv",     # output file
    combined_output,            # the array to save
    delimiter=",",  # comma‐separated
    fmt="%.18e"     # format each float; adjust as needed
    )

    
    # split patoms based on colour threshold
    differences = np.diff(combined_output[:, 0])
    split_indices = np.where(differences > threshold)[0] + 1
    chunks = np.split(combined_output, split_indices)
    
    norm_patoms = []
    for i in chunks:
        center_x = (i.shape[0]-1)/2; center_y = (i.shape[1]-1)/2 # scalar will never change
        num_segments = 16 # scalar will never change
        segment_width = 360 / num_segments
        # skip over and don't save/return patoms that take up 70% or more of the array pixel number
        # only included as my laptop does not have ability to handle large volumes of data
        # if i.shape[0] >= ((single_frame_array.shape[0] * single_frame_array.shape[1]) * 0.7):
        #     pass
        # # # ignore patoms that are less than 2 pixels in size (this is 1/2000 of the input array) (skipping detail but laptop can't proccess otherwise)
        # elif i.shape[0] <= 3:
        #     pass
        # else:
        x_vals = i[:,1]; y_vals = i[:,2]
        
        x_mean = np.floor(x_vals.mean()); y_mean = np.floor(y_vals.mean())
        min_x = x_vals.min(); max_x = x_vals.max(); denominator_x = max_x - min_x
        adj_denom_x = np.where(denominator_x == 0, 1, denominator_x)
        norm_x = 2 * ((x_vals - x_vals.min()) / adj_denom_x) - 1
        
        min_y = y_vals.min(); max_y = y_vals.max(); denominator_y = max_y - min_y
        adj_denom_y = np.where(denominator_y == 0, 1, denominator_y)
        norm_y = 2 * ((y_vals - y_vals.min()) / adj_denom_y) - 1
        
        colours = i[:,0]
        
        angle_deg = (np.degrees(np.arctan2(center_y - y_mean, x_mean - center_x)) + 360) % 360
        angle_clockwise_from_north = (90 - angle_deg) % 360
        segment = angle_clockwise_from_north // segment_width
        
        patom_id = np.random.default_rng().random(dtype=np.float32)

        ## column stack for id, cent_x, cent_y, segment
        first_row = np.array([patom_id, x_mean, y_mean, segment])
        ## column stack for min x, max x, min y, max y
        second_row = np.array([min_x, max_x, min_y, max_y])
        ## column stack for norm x, norm y, colours, ...
        patom_values = np.column_stack((norm_x, norm_y, colours))
        patom_padded = np.full((patom_values.shape[0], 1), np.nan)
        patom_values = np.hstack([patom_values, patom_padded])

        # 4 columns (0, 1, 2, 3)
        # row 1 is id, centroid coordinates and segment
        # row 2 is min and max x and y values for original x and y coordinates in the frame
        # remaining rows are the normalised x and y values and the normalised colour at each coordinate
        patom_array = np.vstack((first_row, second_row, patom_values)).astype('float32')
        norm_patoms.append(patom_array)
    
    return norm_patoms