import numpy as np
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import cv2 as cv

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
    orig_loc_i = orig_loc_i / y_len
    orig_loc_j = orig_loc_j / x_len

    originals = np.column_stack((orig_vals, orig_loc_i, orig_loc_j))
    
    tnn_loc_i = true_indices[0] / y_len
    tnn_loc_j = true_indices[1] / x_len
    tnn_vals = orig_array[true_indices[0], true_indices[1]]
    
    nearest_neigbours = np.column_stack((tnn_vals, tnn_loc_i, tnn_loc_j))

    orig_nn = np.vstack((originals, nearest_neigbours))
    
    return orig_nn

def patoms2d(x_len, y_len, single_frame_array, frame_ind):
    items = [(x_len, y_len, single_frame_array, i) for i in range(8)]
    # with multiprocessing
    atime = perf_counter()
    with Pool(processes=4) as pool:
        res = pool.starmap(snapshot, items)

    # combine the outputs of each nearest neighbour function
    combined_output = np.vstack((res))
    combined_output = np.unique(combined_output[combined_output[:,0].argsort()], axis=1)
    norm_x = 2 * ((combined_output[:,1] - combined_output[:,1].min()) / (combined_output[:,1].max() - combined_output[:,1].min())) - 1 
    norm_y = 2 * ((combined_output[:,2] - combined_output[:,2].min()) / (combined_output[:,2].max() - combined_output[:,2].min())) - 1 
    combined_output = np.column_stack((combined_output[:,0], norm_x, norm_y))
    
    # split patoms based on colour threshold
    differences = np.diff(combined_output[:, 0])
    split_indices = np.where(differences > threshold)[0] + 1
    chunks = np.split(combined_output, split_indices)
    
    norm_patoms = []
    for i in chunks:
        x_vals = i[:,1]; y_vals = i[:,2]
        pat_len = i.shape[0]
        
        pattern_centroid_x = np.array([x_vals.mean()] * pat_len).reshape(pat_len,1)
        pattern_centroid_y = np.array([y_vals.mean()] * pat_len).reshape(pat_len,1)
        frame_ind_arr = np.array([frame_ind] * pat_len).reshape(pat_len,1)
        colours = i[:,0]

        cond_x = [x_vals >= 0, x_vals < 0]
        choice_x = [1, 2]
        quadx = np.select(cond_x, choice_x)
        #print(quadx)
        cond_y = [y_vals >= 0, y_vals < 0]
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
        patom_array = np.column_stack((colours, x_vals, y_vals, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr))
        ind = np.lexsort((patom_array[:,1], patom_array[:,2]))
        patom_array = patom_array[ind]
        
        norm_patoms.append(patom_array)

    #stacked_patoms = np.vstack(norm_patoms)

    print("Time to get 2D patterns with multiprocessing (secs):", (perf_counter()-atime))

    return norm_patoms

## access the camera to get video stream
cap = cv.VideoCapture(0)

val = 0
while val <= 5:
    ret, frame = cap.read()
    # flatten frame and encode into a true colour 24bit integer in RGB format (type uint8?)
    frame = (frame[..., 0] << 16) | (frame[..., 1] << 8) | frame[..., 2]
    frame = (frame - 127.5) / 127.5
    x_len = frame.shape[1]
    y_len = frame.shape[0]
    ####################### FIRST TASK: FIND PATTERNS IN FRAME ######################
    items = [(frame, val)]
    # with multiprocessing
    atime = perf_counter()
    results = patoms2d(x_len, y_len, frame, val)
    frame_patoms_arr = np.vstack((results))
    print(frame_patoms_arr.nbytes)
    # print(frame_patoms[0])
    val += 1