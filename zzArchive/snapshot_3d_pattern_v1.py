# https://stackoverflow.com/questions/70172677/compare-value-in-a-2d-array-to-nearby-values
import numpy as np
from operator import itemgetter
import math
from time import perf_counter
#from memory_profiler import profile
# operating_time = strftime("%H:%M:%S", localtime())
# timestamp = datetime.datetime.now() #.strftime("%Y-%m-%d %H:%M:%S.%f")
# import cv2

# # Import the video and cut it into frames.
# vid = cv2.VideoCapture('/home/gerard/Videos/Webcam/test.webm')

# frames = []
# check = True

# while check:
#     check, arr = vid.read()
#     while arr.all():
#         arr1 = arr[:,:,0]
#         arr2 = arr[:,:,1]
#         arr3 = arr[:,:,2]
#         array = np.concatenate((arr1, arr2, arr3), axis=1)
#         frames.append(array)

# frames = np.array(frames)  # convert list of frames to numpy array
# print(frames.shape)

# from PIL import Image
# #image = Image.open('/home/gerard/Downloads/Plain_Yellow_Star.png')
# #image = Image.open('/home/gerard/Downloads/Golden_star.svg.png')
# image = Image.open('/home/gerard/Downloads/toilet_leak.jpeg')
# numpydata = np.asarray(image)

# time_change = datetime.timedelta(seconds=1/30) 

# list_of_timestamps = []
# for i in range(1, 31):
#     if list_of_timestamps:
#         new_time = list_of_timestamps[-1] + time_change
#         list_of_timestamps.append(new_time)
#     else:
#         list_of_timestamps.append(timestamp)

# timestamps = [t.strftime("%Y-%m-%d %H:%M:%S.%f") for t in list_of_timestamps]

# array = np.random.random((30, 1280, 720))
# @profile
def snapshot_pattern(array):
    strt_stack = perf_counter()
    threshold = 0.00005
    orig_arr = array
    row_len = orig_arr.shape[0]
    col_len = orig_arr.shape[1]
    depth_len = orig_arr.shape[2]
    
    pad_arr = np.pad(array, 1, mode='constant', constant_values=(np.nan)) # memory heavy function 227MiB - consider the need for it
    n1_arr = pad_arr[2:,2:, 2:]; 
    n2_arr = pad_arr[2:,1:col_len+1, 2:]; 
    n3_arr = pad_arr[2:,:col_len, 2:] 
    n4_arr = pad_arr[1:row_len+1,:col_len, 2:]
    n5_arr = pad_arr[:row_len,:col_len, 2:] 
    n6_arr = pad_arr[:row_len,1:col_len+1, 2:]  
    n7_arr = pad_arr[:row_len,2:, 2:]   
    n8_arr = pad_arr[1:row_len+1,2:, 2:]
    n9_arr = pad_arr[1:row_len+1,1:col_len+1, 2:] 

    n10_arr = pad_arr[2:,2:, 1:depth_len+1]; 
    n11_arr = pad_arr[2:,1:col_len+1, 1:depth_len+1]; 
    n12_arr = pad_arr[2:,:col_len, 1:depth_len+1] 
    n13_arr = pad_arr[1:row_len+1,:col_len, 1:depth_len+1] 
    n14_arr = pad_arr[:row_len,:col_len, 1:depth_len+1] 
    n15_arr = pad_arr[:row_len,1:col_len+1, 1:depth_len+1] 
    n16_arr = pad_arr[:row_len,2:, 1:depth_len+1] 
    n17_arr = pad_arr[1:row_len+1,2:, 1:depth_len+1] 

    n18_arr = pad_arr[2:,2:, :depth_len]; 
    n19_arr = pad_arr[2:,1:col_len+1, :depth_len]; 
    n20_arr = pad_arr[2:,:col_len, :depth_len] 
    n21_arr = pad_arr[1:row_len+1,:col_len, :depth_len]
    n22_arr = pad_arr[:row_len,:col_len, :depth_len] 
    n23_arr = pad_arr[:row_len,1:col_len+1, :depth_len]  
    n24_arr = pad_arr[:row_len,2:, :depth_len]   
    n25_arr = pad_arr[1:row_len+1,2:, :depth_len]
    n26_arr = pad_arr[1:row_len+1,1:col_len+1, :depth_len] 


    # only dealing with position of 1st nn (i-1, j-1, k-1)
    truth = abs(orig_arr - n1_arr) <= threshold # heavy not too bad ~27MiB
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k)) # 7.5 MiB
    orig_vals_inds = list(zip(get_orig_vals, loc1)) # 5 MiB
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2])) # 7.5 MiB
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
    tnn_vals_inds = list(zip(get_tnn_vals, loc2)) # 3.1 MiB
    out1 = orig_vals_inds + tnn_vals_inds; # print(out1)

    # only dealing with position of 2nd nn (i, j-1, k-1)
    truth = abs(orig_arr - n2_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k)) # 7.5 MiB
    orig_vals_inds = list(zip(get_orig_vals, loc1)) # 5 MiB
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2])) # 7.5 MiB
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]]) #2.5 MiB
    tnn_vals_inds = list(zip(get_tnn_vals, loc2)) # 3.1 MiB
    out2 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 3rd nn (i+1, j-1)
    truth = abs(orig_arr - n3_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out3 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 4th nn (i+1, j)
    truth = abs(orig_arr - n4_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out4 = orig_vals_inds + tnn_vals_inds
    
    # # only dealing with position of 5th nn (i+1, j+1)
    truth = abs(orig_arr - n5_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out5 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 6th nn (i, j+1)
    truth = abs(orig_arr - n6_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out6 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 7th nn (i-1, j+1)
    truth = abs(orig_arr - n7_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out7 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 8th nn (i-1, j)
    truth = abs(orig_arr - n8_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out8 = orig_vals_inds + tnn_vals_inds
    
    # # only dealing with position of 9th nn (i-1, j)
    truth = abs(orig_arr - n9_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out9 = orig_vals_inds + tnn_vals_inds

    #################################################################
    # # only dealing with position of 10th nn (i-1, j)
    truth = abs(orig_arr - n10_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out10 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 11th nn (i-1, j)
    truth = abs(orig_arr - n11_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out11 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 12th nn (i-1, j)
    truth = abs(orig_arr - n12_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out12 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 9th nn (i-1, j)
    truth = abs(orig_arr - n13_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out13 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 14th nn (i-1, j)
    truth = abs(orig_arr - n14_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out14 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 15th nn (i-1, j)
    truth = abs(orig_arr - n15_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out15 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 16th nn (i-1, j)
    truth = abs(orig_arr - n16_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out16 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 17th nn (i-1, j)
    truth = abs(orig_arr - n17_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out17 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 18th nn (i-1, j)
    truth = abs(orig_arr - n18_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out18 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 19th nn (i-1, j)
    truth = abs(orig_arr - n19_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out19 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 20th nn (i-1, j)
    truth = abs(orig_arr - n20_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out20 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 21th nn (i-1, j)
    truth = abs(orig_arr - n21_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out21 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 22th nn (i-1, j)
    truth = abs(orig_arr - n22_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out22 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 23th nn (i-1, j)
    truth = abs(orig_arr - n23_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out23 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 24th nn (i-1, j)
    truth = abs(orig_arr - n24_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out24 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 25th nn (i-1, j)
    truth = abs(orig_arr - n25_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out25 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 26th nn (i-1, j)
    # identify array values that when compared to the array value of interest fall within a threshold difference
    truth = abs(orig_arr - n26_arr) <= threshold
    # obtain the indices of each element of the array that is non-zero based on the thresholde truth array above
    true_indices = np.asarray(truth).nonzero()
    # functions to identify the original location of the element being compared against the element of interest
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x-1
    # get the indices of the original location for the array elements being compared against the element of interest
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    # obtain the values connected to these index locations
    get_orig_vals = orig_arr[orig_loc_i, orig_loc_j, orig_loc_k]
    # create a list of original indices
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    # create a list of the original values along with their indices
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    # create a list of the indices for the nearest neighbours who fall within the threshold value
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    # get and create a list of the values associated 
    get_tnn_vals = list(orig_arr[true_indices[0], true_indices[1], true_indices[2]])
    # create a list of the nearest eighbour values and indices
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    # combine all the original values with their indices to the identified nearest neighbour values and indices
    # that fall within the threshold value
    out26 = orig_vals_inds + tnn_vals_inds

    # combine the outputs of each nearest neighbour function
    # how? concat all the lists, then order by value, split list when difference between
    # 2 lists elements is larger than threshold, re-order split lists by indices
    # these final split re-order lists are the patterns I am lookng for
    outs = sorted(set(out1+out2+out3+out4+out5+out6+out7+out8+ 
                      out9+out10+out11+out12+out13+out14+out15+out16+ 
                      out10+out11+out12+out13+out14+out15+out16+out17+ 
                      out18+out19+out20+out21+out22+out23+out24+out25+out26
                      )) # heavy ish weird but ok 24 MiB
    
    # split list when value between subsequent elements is greater than threshold
    res, last = [[]], None
    for x in outs:
        if last is None or abs(last - x[0]) <= threshold: #runtime warning here
            res[-1].append(x)
        else:
            res.append([x])
        last = x[0]

    # sort the lists of tuples based on the indices (need to get indices as tuple)
    s_res = []
    for i in res:
        s = sorted(i, key=itemgetter(1))
        if len(s) >= 10:
            s_res.append(s)

    # then need to obtain a normalised distance for all points from the 'center' of the pattern
    norm_patoms = []
    for ind, pat in enumerate(s_res):
        x = [p[1][0] for p in pat]
        min_x = min(x); max_x = max(x)
        norm_x = [(i - min_x)/(max_x - min_x) for i in x]
        y = [p[1][1] for p in pat]
        min_y = min(y); max_y = max(y)
        norm_y = [(i - min_y)/(max_y - min_y) for i in y]
        z = [p[1][2] for p in pat]
        min_z = min(z); max_z = max(z)
        norm_z = [(i - min_z)/(max_z - min_z) for i in z]
        centroid = list((sum(x) / len(pat), sum(y) / len(pat), sum(z) / len(pat)))
        centroid_norm = list((sum(norm_x) / len(pat), sum(norm_y) / len(pat), sum(norm_z) / len(pat)))
        centroid_list = [tuple([0.0]+centroid+centroid_norm+[0.0])]
        loc = [p[1] for p in pat]
        dist = list(map(lambda x: math.dist(centroid, list(x)), loc))
        tot = sum(dist)
        # normalised distance value is from centroid
        norm_dist = [i/tot for i in dist]
        val = [p[0] for p in pat]
        patom = list(zip(val, x, y, z, norm_x, norm_y, norm_z, norm_dist))
        patom = patom + centroid_list
        norm_patoms.append(patom)

    end_stack = perf_counter()

    stacked_time = end_stack - strt_stack
    print('Took this many seconds: ', stacked_time)

    return norm_patoms