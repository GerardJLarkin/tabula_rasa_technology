# function to create actual reference patom from group members
import numpy as np

def split_rows(arr, avg_rows):
 
    array_shape = arr.shape[0]
    
    # id the shape of the indivdiaul patom is greater or equal to the average number of rows
    # split the array into subset arrays based on the number of rows 
    if array_shape >= avg_rows:
        # caluclate number of subset arrays (floor to get integer number of subsets)
        subsets = array_shape // avg_rows
        # get modulo of array shape and avg number of rows to determine in rows need to be allocated
        # to more than one subset
        mod  = array_shape % avg_rows  
        # create array to allow subset end values to be determined (so we can split the patom array correctly based on row index)
        sizes = np.full(avg_rows, subsets, dtype=int)
        
        if mod != 0:
            # find index locations to add an extra row to distribute the add'n rows evenly across the subsets
            extra_pos = np.floor(np.arange(mod) * avg_rows / mod).astype(int)
            sizes[extra_pos] += 1
        
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        
        return [arr[s:e] for s, e in zip(starts, starts + sizes)]

    # if the patom array has a smaller shape than the average rows, repeat rows evenly across the subsets
    idx = np.floor(np.arange(avg_rows) * array_shape / avg_rows).astype(int)
    
    return [arr[i:i+1] for i in idx]

def create_reference_patom(arrays):

    ## added to allow for testing of refernce creation
    group_first_rows = np.vstack([pat[[0],:] for pat in arrays])
    group_first_rows = np.nanmean(group_first_rows, axis=0, keepdims=True)

    # get second rows to find average min_x, max_x, min_y, max_y
    group_second_rows = np.vstack([pat[[1],:] for pat in arrays])
    group_second_rows = np.nanmean(group_second_rows, axis=0, keepdims=True)

    group_values = np.vstack([pat[2:,:3] for pat in arrays])
    avg_rows = int(np.floor(group_values.shape[0] / len(arrays))) # is it better to have a higher or lower number of avg rows?
    
    # create an array for each row in the average number of rows
    # this is to hold row(s) corresponding to the split peformed on the patoms 
    split_arrays = [[] for _ in range(avg_rows)]
    
    # split each patom into subsections, then rejoin corresponding subsections to calulcate average values 
    # for each section and then stitch these back together to create a reference patom
    for i in arrays:
        arr = i[2:,:3]
        subarrays = split_rows(arr, avg_rows)
        # for each set of 'rows' in the subarray, sum along axis 0 and divide by the number of rows in the split
        for idx, j in enumerate(subarrays):
            first_sum = np.sum(j, axis=0, keepdims=True)
            split_arrays[idx].append(first_sum)

    
    stacked_split_arrays = []
    for row_array in split_arrays:
        stacked_rows = np.vstack((row_array))
        second_sum = np.sum(stacked_rows, axis=0, keepdims=True) / stacked_rows.shape[0]
        stacked_split_arrays.append(second_sum)
    
    reference_array_values = np.vstack((stacked_split_arrays))

    ref_patom_values = np.hstack((reference_array_values, np.full((reference_array_values.shape[0],1), np.nan)))

    ref_id = np.random.default_rng().random(dtype=np.float32)

    # reference_patom = np.vstack((
    #     np.array([[ref_id, np.nan, np.nan, np.nan]]),
    #     group_second_rows, 
    #     ref_patom_values))
    
    ## added for testing of group creation
    reference_patom = np.vstack((
        np.array([[ref_id, group_first_rows[0,1], group_first_rows[0,2], np.nan]]),
        group_second_rows, 
        ref_patom_values))

    return reference_patom