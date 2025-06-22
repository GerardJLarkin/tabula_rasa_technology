# https://stackoverflow.com/questions/70172677/compare-value-in-a-2d-array-to-nearby-values
import numpy as np

def slidingCompare(arr, footprint=(3, 3), threshold=0.05):
    """
          arr: 2D array   | input
    footprint: tuple      | search window dimensions (must be odd)
    threshold: float      | Threshold for neighbours to be close
    """

    assert footprint[0] % 2 == 1, "Footprint dimensions should be odd. "
    assert footprint[0] % 2 == 1, "Footprint dimensions should be odd. "
    
    # takes in the array, gets the shape along each axis adds the footprint shape on the same axis and substracts 1 
    # this returns an array that is padded with an extra column
    temp_arr = np.full((arr.shape[0] + footprint[0] - 1, 
                        arr.shape[1] + footprint[1] - 1), np.nan)
    
    # seems to return the original input array so I'm not sure what its doing
    temp_arr[footprint[0] // 2:footprint[0] // 2 + arr.shape[0],
             footprint[1] // 2:footprint[1] // 2 + arr.shape[1]] = arr
    
    # returns arrays for the row and col indices
    i_all, j_all = np.mgrid[-(footprint[0] // 2):arr.shape[0]+(footprint[0] // 2), 
                            -(footprint[1] // 2):arr.shape[1]+(footprint[1] // 2)]
    
    # Footprint around the current element (ie looking at the 8 elements around the central value). Must be odd.
    footprint_size = np.prod(footprint)
    
    # creates nan array where we can insert output for i and j indices
    output_i = np.full((footprint_size, *arr.shape), np.nan)
    output_j = np.full((footprint_size, *arr.shape), np.nan)
    
    # create 3x3 array for what reason?
    output_ix = np.arange(footprint_size).reshape(footprint)
    
    # loop through vertical positions on the footprint size array
    for vert_pos in np.arange(footprint[0]):
        #print(vert_pos)
        # loop through horizontal positions on the footprint size array
        for horiz_pos in np.arange(footprint[1]):
            #print(horiz_pos)
            # create arrays that are shifted by the vert and horiz positions
            neighbour = temp_arr[vert_pos: vert_pos + arr.shape[0], 
                                 horiz_pos: horiz_pos + arr.shape[1]]
            # for each of the previously created arrays substract the neighbours from the original array,
            # and return if they are below the threshold
            close_mask = abs(arr - neighbour) <= threshold
            print(vert_pos, horiz_pos)
            print(arr)
            print(neighbour)
            print(close_mask)
            # update the previously create indice based nan arrays, with the value of the neighbour for each index
            # that falls within the threshold value
            #print(output_ix[vert_pos, horiz_pos], close_mask)
            #print(i_all[vert_pos: vert_pos + arr.shape[0], horiz_pos: horiz_pos + arr.shape[1]][close_mask])
            output_i[output_ix[vert_pos, horiz_pos], close_mask] = i_all[vert_pos: vert_pos + arr.shape[0], 
                                                    horiz_pos: horiz_pos + arr.shape[1]][close_mask]
            #print(output_i)
            output_j[output_ix[vert_pos, horiz_pos], close_mask] = j_all[vert_pos: vert_pos + arr.shape[0], 
                                                    horiz_pos: horiz_pos + arr.shape[1]][close_mask]
            
    # Output: two 3D arrays of indices corresponding to elements within the threshold of the element of interest for rows and cols
    return output_i, output_j

#np.random.seed(42)
array = np.random.random((4, 3))
print(array)

#(slidingCompare(array, footprint=(3, 3), threshold=0.5))

# what do I want? 
# I want to take a single pixel value and compare it to its 8 nearest neighbours
# if a nearest neighbour falls within a threshold value, I want to store the original pixel value and its indices
# along with the neighbour pixel values and it indices. this will result in 8 tuples? the first element of the tuple
# is the pixl value, the second element is another tuple of the indices
# the 8 tuples will then be grouped and the tuples will be combined to form a new tuple/list with the pixel values in
# order of the indices (i, j matrix schema)
# pixel starts at (i,j)
# 1st neighbour (i-1, j-1)
# 2nd neighbour (i, j-1)
# 3rd neighbour (i+1, j-1)
# 4th neighbour (i+1, j)
# 5th neighbour (i+1, j+1)
# 6th neighbour (i, j+1)
# 7th neighbour (i-1, j+1)
# 8th neighbour (i-1, j)
# we want to perform a vector operation NO LOOPING through array

def nearestNeighbourCompare(array):
    threshold = 0.1
    orig_arr = array
    row_len = orig_arr.shape[0]
    col_len = orig_arr.shape[1]
    
    pad_arr = np.pad(array, 1, mode='constant', constant_values=(np.nan)) # pad array on all sides with nans
    n1_arr = pad_arr[2:,2:]  # shift array up by 1, down by 0 and right by 0, left by 1 (1st n)   
    n2_arr = pad_arr[2:,1:col_len+1] # shift array up by 1, down by 0 and right by 0, left by 0 (2nd n) 
    n3_arr = pad_arr[2:,:col_len] # shift array up by 1, down by 0 and right by 1, left by 1 (3rd n) 
    n4_arr = pad_arr[1:row_len+1,:col_len]# shift array up by 0, down by 0 and right by 1, left by 0 (4th n)  
    n5_arr = pad_arr[:row_len,:col_len] # shift array up by 0, down by 1 and right by 1, left by 0 (5th n) 
    n6_arr = pad_arr[:row_len,1:col_len+1]  # shift array up by 0, down by 1 and right by 0, left by 0 (6th n)  
    n7_arr = pad_arr[:row_len,2:] # shift array up by 0, down by 1 and right by 0, left by 1 (7th n)   
    n8_arr = pad_arr[1:row_len+1,2:] # shift array up by 0, down by 0 and right by 0, left by 1 (8th n) 
    
    # only dealing with position of 1st nn (i-1, j-1)
    truth = abs(orig_arr - n1_arr) <= threshold
    #print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    #print(orig_loc_i, orig_loc_j)
    #print(true_indices[0], true_indices[1])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out1 = orig_vals_inds + tnn_vals_inds
    #print(out1)

    # only dealing with position of 2nd nn (i, j-1)
    truth = abs(orig_arr - n2_arr) <= threshold
    #print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]) # appears getting the true indices returns i, j inverted
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    # print(orig_loc_i, orig_loc_j)
    # print(true_indices[0], true_indices[1])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out2 = orig_vals_inds + tnn_vals_inds
    #print(out2)

    # only dealing with position of 3rd nn (i+1, j-1)
    truth = abs(orig_arr - n3_arr) <= threshold
    # print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    # print(orig_loc_i, orig_loc_j)
    # print(true_indices[0], true_indices[1])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out3 = orig_vals_inds + tnn_vals_inds
    # print(out3)

    # only dealing with position of 4th nn (i+1, j)
    truth = abs(orig_arr - n4_arr) <= threshold
    # print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    # print('orig', orig_loc_i, orig_loc_j)
    # print('true', true_indices[0], true_indices[1])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    # print('loc1', loc1)
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    # print('orig_vals', orig_vals_inds)
    loc2 = list(zip(true_indices[0], true_indices[1]))
    # print('loc2', loc2)
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    # print('true_vals', tnn_vals_inds)
    out4 = orig_vals_inds + tnn_vals_inds
    # print(out4)
    
    # only dealing with position of 5th nn (i+1, j+1)
    truth = abs(orig_arr - n5_arr) <= threshold
    # print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out5 = orig_vals_inds + tnn_vals_inds
    # print(out5)

    # only dealing with position of 6th nn (i, j+1)
    truth = abs(orig_arr - n6_arr) <= threshold
    # print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out6 = orig_vals_inds + tnn_vals_inds
    # print(out6)

    # only dealing with position of 7th nn (i-1, j+1)
    truth = abs(orig_arr - n7_arr) <= threshold
    # print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    get_orig_vals = list(orig_arr[orig_loc_i, orig_loc_j])
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out7 = orig_vals_inds + tnn_vals_inds
    # print(out7)

    # only dealing with position of 8th nn (i-1, j)
    truth = abs(orig_arr - n8_arr) <= threshold
    # print(truth)
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    get_orig_vals = list(orig_arr[orig_loc_i, orig_loc_j])
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out8 = orig_vals_inds + tnn_vals_inds
    # print(out8)

    # combine the outputs of each nearest neighbour function
    # how? concat all the lists, then order by value, split list when difference between
    # 2 lists elements is larger than threshold, re-order split lists by indices
    # these final split re-order lists are the patterns I am lookng for
    outs = sorted(set(out1+out2+out3+out4+out5+out6+out7+out8))
    # sorted_outs = sorted(outs)
    print(outs)
    # res = [outs[i + 1][0] - outs[i][0] for i in range(len(outs) - 1)]
    # print(res)

    res, last = [[]], None
    for x in outs:
        if last is None or abs(last - x[0]) <= 0.1:
            res[-1].append(x)
        else:
            res.append([x])
        last = x[0]

    print(res)

nearestNeighbourCompare(array)