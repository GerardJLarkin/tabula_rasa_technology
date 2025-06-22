# https://stackoverflow.com/questions/70172677/compare-value-in-a-2d-array-to-nearby-values
import numpy as np
from operator import itemgetter
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import cv2 as cv
from numba import njit, prange
import numba

# from PIL import Image
# #image = Image.open('/home/gerard/Downloads/Plain_Yellow_Star.png')
# #image = Image.open('/home/gerard/Downloads/Golden_star.svg.png')
# image = Image.open('/home/gerard/Downloads/toilet_leak.jpeg')
# numpydata = np.asarray(image)

# array = numpydata[:,:,0]
# print(array.shape)

# # array = np.random.random((40, 30))
# # print(array)

def snapshot(array):
    threshold = 3.99
    orig_arr = array
    row_len = orig_arr.shape[0]; print(row_len)
    col_len = orig_arr.shape[1]; print(col_len)
    
    pad_arr = np.pad(array, 1, mode='constant', constant_values=(np.nan)) # pad array on all sides with nans
    # print(pad_arr.shape)
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
    true_indices = np.asarray(truth).nonzero()
    # print(truth.shape)
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j]
    loc1 = list(zip(orig_loc_i, orig_loc_j))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out1 = orig_vals_inds + tnn_vals_inds

    # only dealing with position of 2nd nn (i, j-1)
    truth = abs(orig_arr - n2_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
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
    out2 = orig_vals_inds + tnn_vals_inds

    # only dealing with position of 3rd nn (i+1, j-1)
    truth = abs(orig_arr - n3_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
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
    out3 = orig_vals_inds + tnn_vals_inds

    # only dealing with position of 4th nn (i+1, j)
    truth = abs(orig_arr - n4_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
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
    out4 = orig_vals_inds + tnn_vals_inds
    
    # only dealing with position of 5th nn (i+1, j+1)
    truth = abs(orig_arr - n5_arr) <= threshold
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

    # only dealing with position of 6th nn (i, j+1)
    truth = abs(orig_arr - n6_arr) <= threshold
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

    # only dealing with position of 7th nn (i-1, j+1)
    truth = abs(orig_arr - n7_arr) <= threshold
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

    # only dealing with position of 8th nn (i-1, j)
    truth = abs(orig_arr - n8_arr) <= threshold
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

    # combine the outputs of each nearest neighbour function
    # how? concat all the lists, then order by value, split list when difference between
    # 2 lists elements is larger than threshold, re-order split lists by indices
    # these final split re-order lists are the patterns I am lookng for
    outs = sorted(set(out1+out2+out3+out4+out5+out6+out7+out8))
    # split list when value between subsequent elements is greater than threshold
    res, last = [[]], None
    for x in outs:
        if last is None or abs(last - x[0]) <= threshold: #runtime warning here
            res[-1].append(x)
        else:
            res.append([x])
        last = x[0]

    # sort the lists of tuples based on the indices
    s_res = []
    for i in res:
        s = sorted(i, key=itemgetter(1))
        s_res.append(s)
    #print(s_res)

    # then need to obtain a normalised distance for all points from the 'center' of the pattern
    norm_dist_patoms = []
    for pat in s_res:
        x = [p[1][0] for p in pat]
        y = [p[1][1] for p in pat]
        centroid = list((sum(x) / len(pat), sum(y) / len(pat)))
        loc = [p[1] for p in pat]
        dist = list(map(lambda x: math.dist(centroid, list(x)), loc))
        tot = sum(dist)
        norm = [i/tot for i in dist]
        val = [p[0] for p in pat]
        patom = list(zip(val, norm))
        norm_dist_patoms.append(patom)

    #print(norm_dist_patoms)

    return norm_dist_patoms



# identified_patterns = snapshot_pattern(array)

# print(len(identified_patterns))

# for ind, i in enumerate(identified_patterns):
#     # Determine the shape of the resulting array
#     # Assuming that the indices cover the full range of the array
#     shape = array.shape

#     # # Initialize an empty NumPy array with the desired shape
#     array = np.zeros(shape, dtype=int)
#     counter = 0
#     # Fill the array with the values from the list
#     for value, index in i:
#         array[index] = value
#         counter += 1
    
#     plt.imshow(array)
#     plt.axis('off')
#     plt.savefig('pattern_{ind}_{counter}.png'.format(ind, counter))

## access the camera to get video stream
cap = cv.VideoCapture(0)

val = 0
while val <= 5:
    ret, frame = cap.read()
    # flatten frame and encode into a true colour 24bit integer in RGB format (type uint8?)
    frame = (frame[..., 0] << 16) | (frame[..., 1] << 8) | frame[..., 2]
    frame = (frame - 127.5) / 127.5
    print(frame)
    ####################### FIRST TASK: FIND PATTERNS IN FRAME ######################
    # frame_patoms = patoms2d(frame, val)
    # print(frame_patoms[0])
    # items = [(frame, i) for i in range(8)]
    # with multiprocessing
    atime = perf_counter()
    with Pool(processes=4) as pool:
        res = pool.map(snapshot, frame)
        print("Time to get 2D patterns with multiprocessing (secs):", (perf_counter()-atime))
    val += 1