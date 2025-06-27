import numpy as np
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME

threshold = 0.0005 #0.00005 -- need to reasses this when we get live data

def snapshot(single_frame_array):
    orig_array = single_frame_array
    print('rows',orig_array.shape[0])
    print('cols',orig_array.shape[1])

    # Step 1: Compute differences along rows and columns
    row_diff = np.diff(orig_array, axis=1)  # Differences along rows
    col_diff = np.diff(orig_array, axis=0)  # Differences along columns

    # Step 2: Identify where differences exceed the threshold
    row_split_mask = row_diff > threshold  # Mask for row splits
    col_split_mask = col_diff > threshold  # Mask for column splits

    # Step 3: Generate indices for rows and columns
    row_indices = np.arange(orig_array.shape[1])  # Column indices
    col_indices = np.arange(orig_array.shape[0])  # Row indices

    # Step 4: Create cumulative group labels for rows and columns
    row_groups = np.cumsum(np.pad(row_split_mask, ((0, 0), (1, 0)), constant_values=True), axis=1)
    col_groups = np.cumsum(np.pad(col_split_mask, ((1, 0), (0, 0)), constant_values=True), axis=0)

    # Step 5: Combine row and column group labels
    combined_groups = row_groups + col_groups * (orig_array.shape[1] + 1)

    # Step 6: Extract unique group labels and their indices
    unique_groups, group_indices = np.unique(combined_groups, return_inverse=True)

    # Step 7: Reshape group indices to match the array shape
    group_indices = group_indices.reshape(orig_array.shape)

    # Step 8: Group values and indices
    grouped_values = [orig_array[group_indices == group] for group in unique_groups]
    grouped_indices = [np.argwhere(group_indices == group) for group in unique_groups]

    return grouped_values, grouped_indices

def patoms2d(single_frame_array):
    # items = [(single_frame_array, i) for i in range(8)]
    # with multiprocessing
    res = snapshot(single_frame_array)
    atime = perf_counter()
    # with Pool(processes=cpu_count()) as pool:
    #     res = pool.map(snapshot, single_frame_array) # tuple output: groups of values, groups of indices

    # then need to obtain a normalised distance for all points from the 'center' of the pattern
    # norm_patoms = []
    # for patom_ind, pat in enumerate(s_res):
    #     pat_len = len(pat)
    #     x_vals = [p[1][0] for p in pat]; min_x = min(x_vals); max_x = max(x_vals)
    #     norm_x = np.array([2 * (x - min_x) / (max_x - min_x) - 1 for x in x_vals]).reshape(pat_len,1)
    #     y_vals = [p[1][1] for p in pat]; min_y = min(y_vals); max_y = max(y_vals)
    #     norm_y = np.array([2 * (x - min_y) / (max_y - min_y) - 1 for x in y_vals]).reshape(pat_len,1)
    #     pattern_centroid_x = np.array([sum(norm_x)/pat_len] * pat_len).reshape(pat_len,1)
    #     pattern_centroid_y = np.array([sum(norm_y)/pat_len] * pat_len).reshape(pat_len,1)
    #     patom_ind = np.array([patom_ind] * pat_len).reshape(pat_len,1)
    #     frame_ind_arr = np.array([frame_ind] * pat_len).reshape(pat_len,1)
        
    #     patom_time = np.array([clock_gettime_ns(CLOCK_REALTIME)] * pat_len).reshape(pat_len,1)

    #     cond_x = [norm_x >= 0, norm_x < 0]
    #     choice_x = [1, 2]
    #     quadx = np.select(cond_x, choice_x)
    #     cond_y = [norm_y >= 0, norm_y < 0]
    #     choice_y = [3, 4]
    #     quady = np.select(cond_y, choice_y)
    #     quad = np.hstack([quadx, quady])
    #     quad = np.array([int(f"{a}{b}") for a, b in quad]).reshape(pat_len,1)

    #     # Get unique values and their counts
    #     unique_values, counts = np.unique(quad, return_counts=True)
    #     # Create a dictionary to map values to their counts
    #     value_to_count = {value: count for value, count in zip(unique_values, counts)}
    #     # Add a new column with the counts
    #     quad_cnt = np.array([value_to_count[value] for value in quad.flatten()]).reshape(pat_len, 1)
    #     # 9 columns (0,1,2,3,4,5,6,7,8)
    #     patom_array = np.hstack([norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, patom_ind, frame_ind_arr, patom_time]).astype(np.float32)
        
    #     norm_patoms.append(patom_array)

    #stacked_patoms = np.vstack(norm_patoms)

    print("Time to get 2D patterns with multiprocessing (secs):", (perf_counter()-atime))

    return res # norm_patoms