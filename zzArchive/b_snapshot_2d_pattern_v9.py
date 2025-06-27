import numpy as np
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import perf_counter, process_time, clock_gettime_ns, CLOCK_REALTIME

threshold = 0.00005 #0.00005 -- need to reasses this when we get live data
# motion = [[-1, -1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]
motion = np.array([[-1, -1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]])

def snapshot(x_len, y_len, single_frame_array, i):

    # # Extract motion indices
    # dx, dy = motion[i]
    
    # # Define slicing indices in a single step
    # ia, ib = (1, None) if dx == -1 else (None, -1) if dx == 1 else (None, None)
    # ja, jb = (1, None) if dy == -1 else (None, -1) if dy == 1 else (None, None)

    # xa, xb = (None, -1) if dx == -1 else (1, None) if dx == 1 else (None, None)
    # ya, yb = (None, -1) if dy == -1 else (1, None) if dy == 1 else (None, None)

    # # Extract sub-arrays using optimized slicing
    # orig_array = single_frame_array[1:-1, 1:-1]  # Remove border
    # arr = orig_array[ia:ib, ja:jb]
    # comp = orig_array[xa:xb, ya:yb]

    # # Compute element-wise truth array
    # truth = np.abs(comp - arr) <= threshold
    # true_indices = np.nonzero(truth)

    # # Compute original locations in a single step
    # orig_loc_i = true_indices[0] + dx
    # orig_loc_j = true_indices[1] + dy

    # # Extract values efficiently
    # orig_vals = orig_array[orig_loc_i, orig_loc_j]
    # tnn_vals = orig_array[true_indices]

    # # Combine results using optimized stacking
    # originals = np.column_stack((orig_vals, orig_loc_i, orig_loc_j))
    # nearest_neighbours = np.column_stack((tnn_vals, true_indices[0], true_indices[1]))

    # return np.vstack((originals, nearest_neighbours))


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

def patoms2d(x_len, y_len, single_frame_array, frame_ind):
    items = [(x_len, y_len, single_frame_array, i) for i in range(8)]
    # with multiprocessing
    atime = perf_counter(), process_time()
    with Pool(processes=2) as pool:
        res = pool.starmap(snapshot, items)

    # combine the outputs of each nearest neighbour function
    combined_output = np.vstack((res))
    combined_output = np.unique(combined_output, axis=0)
    combined_output = combined_output[combined_output[:,0].argsort()]
    
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

    #stacked_patoms = np.vstack(norm_patoms)

    print("Real Time to get 2D patterns with multiprocessing (secs):", (perf_counter()-atime[0]))
    print("CPU Time to get 2D patterns with multiprocessing (secs):", (process_time()-atime[1]))

    return norm_patoms