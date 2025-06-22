# https://stackoverflow.com/questions/70172677/compare-value-in-a-2d-orig_array-to-nearby-values
import numpy as np
from operator import itemgetter
import math
from time import perf_counter

np.random.seed(5555)
rand_array = np.random.random((30, 720, 1280))

def snapshot_pattern(array):
    strt_stack = perf_counter()
    threshold = 0.00005
    orig_array = array
    d = orig_array.shape[0]
    h = orig_array.shape[1]
    w = orig_array.shape[2]
    
    n1_arr = orig_array[1:,1:,1:]
    n1_comp = orig_array[:-1,:-1,:-1]
    #[-1, -1, -1]

    n2_arr = orig_array[1:,1:,:]
    n2_comp = orig_array[:-1,:-1,:]
    #[-1, -1,  0]
    
    n3_arr = orig_array[1:,1:,:-1]
    n3_comp = orig_array[:-1,:-1,1:]
    #[-1, -1,  1]
    
    n4_arr = orig_array[1:,:,1:]
    n4_comp = orig_array[:-1,:,:-1]
    #[-1,  0, -1]

    n5_arr = orig_array[1:,:,:]
    n5_comp = orig_array[:-1,:,:]
    #[-1,  0,  0]
     
    n6_arr = orig_array[1:,:,:-1]
    n6_comp = orig_array[:-1,:,1:]
    #[-1,  0,  1]

    n7_arr = orig_array[1:,:-1,1:] 
    n7_comp = orig_array[:-1,1:,:-1]
    #[-1,  1, -1]
    
    n8_arr = orig_array[1:,:-1,:]
    n8_comp = orig_array[:-1,1:,:]
    #[-1,  1,  0]
    
    n9_arr = orig_array[1:,:-1,:-1]
    n9_comp = orig_array[:-1,1:,1:]
    #[-1,  1,  1]

    n10_arr = orig_array[:,1:,1:]
    n10_comp = orig_array[:,:-1,:-1]
    #[ 0, -1, -1]
    
    n11_arr = orig_array[:,1:,:] 
    n11_comp = orig_array[:,:-1,:] 
    #[ 0, -1,  0]
    
    n12_arr = orig_array[:,1:,:-1] 
    n12_comp = orig_array[:,:-1,1:] 
    #[ 0, -1,  1]
    
    n13_arr = orig_array[:,:,1:] 
    n13_comp = orig_array[:,:,:-1] 
    #[ 0,  0, -1]
    
    n14_arr = orig_array[:,:,:-1] 
    n14_comp = orig_array[:,:,1:] 
    #[ 0,  0,  1]
    
    n15_arr = orig_array[:,:-1,1:] 
    n15_comp = orig_array[:,1:,:-1]
    #[ 0,  1, -1]
    
    n16_arr = orig_array[:,:-1,:] 
    n16_comp = orig_array[:,1:,:]
    #[ 0,  1,  0]
    
    n17_arr = orig_array[:,:-1,:-1] 
    n17_comp = orig_array[:,1:,1:] 
    #[ 0,  1,  1]

    n18_arr = orig_array[:-1,1:,1:] 
    n18_comp = orig_array[1:,:-1,:-1] 
    #[ 1, -1, -1]
    
    n19_arr = orig_array[:-1,1:,:] 
    n19_comp = orig_array[1:,:-1,:] 
    #[ 1, -1,  0]
    
    n20_arr = orig_array[:-1,1:,:-1] 
    n20_comp = orig_array[1:,:-1,1:] 
    #[ 1, -1,  1]
    
    n21_arr = orig_array[:-1,:,1:]
    n21_comp = orig_array[1:,:,:-1]
    #[ 1,  0, -1]
    
    n22_arr = orig_array[:-1,:,:] 
    n22_comp = orig_array[1:,:,:] 
    #[ 1,  0,  0]
    
    n23_arr = orig_array[:-1,:,:-1]
    n23_comp = orig_array[1:,:,1:]
    #[ 1,  0,  1]  
    
    n24_arr = orig_array[:-1,:-1,1:]  
    n24_comp = orig_array[1:,1:,:-1] 
    #[ 1,  1, -1] 
    
    n25_arr = orig_array[:-1,:-1,:]
    n25_comp = orig_array[1:,1:,:]
    #[ 1,  1,  0]
    
    n26_arr = orig_array[:-1,:-1,:-1] 
    n26_comp = orig_array[1:,1:,1:] 
    #[ 1,  1,  1] 


    # only dealing with position of 1st nn (i-1, j-1, k-1)
    truth = abs(n1_comp - n1_arr) <= threshold # heavy not too bad ~27MiB
    true_indices = np.asarray(truth).nonzero()
    #[-1, -1, -1]
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x+1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))# 7.5 MiB
    orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2])) # 7.5 MiB
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))# 3.1 MiB
    out1 = orig_vals_inds + tnn_vals_inds

    # only dealing with position of 2nd nn (i, j-1, k-1)
    truth = abs(n2_comp - n2_arr) <= threshold # heavy not too bad ~27MiB
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x
    #[-1, -1,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k)); # 7.5 MiB
    orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2])) # 7.5 MiB
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
    tnn_vals_inds = list(zip(get_tnn_vals, loc2)) # 3.1 MiB
    out2 = orig_vals_inds + tnn_vals_inds

    # only dealing with position of 3rd nn (i+1, j-1)
    truth = abs(n3_comp - n3_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x-1
    #[-1, -1,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out3 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 4th nn (i+1, j)
    truth = abs(n4_comp - n4_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x+1
    #[-1,  0, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out4 = orig_vals_inds + tnn_vals_inds
    
    # only dealing with position of 5th nn (i+1, j+1)
    truth = abs(n5_comp - n5_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x
    #[-1,  0,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out5 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 6th nn (i, j+1)
    truth = abs(n6_comp - n6_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x-1
    # [-1,  0,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out6 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 7th nn (i-1, j+1)
    truth = abs(n7_comp - n7_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x+1
    #[-1,  1, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out7 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 8th nn (i-1, j)
    truth = abs(n8_comp - n8_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x
    #[-1,  1,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out8 = orig_vals_inds + tnn_vals_inds
    
    # # only dealing with position of 9th nn (i-1, j)
    truth = abs(n9_comp - n9_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x-1
    #[-1,  1,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out9 = orig_vals_inds + tnn_vals_inds

    # only dealing with position of 10th nn (i-1, j)
    truth = abs(n10_comp - n10_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x+1
    #[ 0, -1, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out10 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 11th nn (i-1, j)
    truth = abs(n11_comp - n11_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x
    #[ 0, -1,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out11 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 12th nn (i-1, j)
    truth = abs(n12_comp - n12_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x-1
    #[ 0, -1,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out12 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 9th nn (i-1, j)
    truth = abs(n13_comp - n13_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x+1
    #[ 0,  0, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out13 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 14th nn (i-1, j)
    truth = abs(n14_comp - n14_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x-1
    #[ 0,  0,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out14 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 15th nn (i-1, j)
    truth = abs(n15_comp - n15_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x+1
    #[ 0,  1, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out15 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 16th nn (i-1, j)
    truth = abs(n16_comp - n16_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x
    #[ 0,  1,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out16 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 17th nn (i-1, j)
    truth = abs(n17_comp - n17_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x-1
    #[ 0,  1,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out17 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 18th nn (i-1, j)
    truth = abs(n18_comp - n18_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x+1
    #[ 1, -1, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out18 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 19th nn (i-1, j)
    truth = abs(n19_comp - n19_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x
    #[ 1, -1,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out19 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 20th nn (i-1, j)
    truth = abs(n20_comp - n20_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x-1
    #[ 1, -1,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out20 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 21th nn (i-1, j)
    truth = abs(n21_comp - n21_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x+1
    #[ 1,  0, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out21 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 22th nn (i-1, j)
    truth = abs(n22_comp - n22_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x
    #[ 1,  0,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out22 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 23th nn (i-1, j)
    truth = abs(n23_comp - n23_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x
    def get_orig_loc_k(x):
        return x-1
    #[ 1,  0,  1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out23 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 24th nn (i-1, j)
    truth = abs(n24_comp - n24_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x+1
    #[ 1,  1, -1]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out24 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 25th nn (i-1, j)
    truth = abs(n25_comp - n25_arr) <= threshold
    true_indices = np.asarray(truth).nonzero()
    def get_orig_loc_i(x):
        return x+1
    def get_orig_loc_j(x):
        return x+1
    def get_orig_loc_k(x):
        return x
    #[ 1,  1,  0]
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))
    out25 = orig_vals_inds + tnn_vals_inds

    # # only dealing with position of 26th nn (i-1, j)
    # identify orig_array values that when compared to the orig_array value of interest fall within a threshold difference
    truth = abs(n26_comp - n26_arr) <= threshold
    # obtain the indices of each element of the orig_array that is non-zero based on the thresholde truth orig_array above
    true_indices = np.asarray(truth).nonzero()
    # functions to identify the original location of the element being compared against the element of interest
    def get_orig_loc_i(x):
        return x-1
    def get_orig_loc_j(x):
        return x-1
    def get_orig_loc_k(x):
        return x-1
    #[ 1,  1,  1]
    # get the indices of the original location for the orig_array elements being compared against the element of interest
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0]); # print(orig_loc_i)
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    # obtain the values connected to these index locations
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    # create a list of original indices
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))
    # create a list of the original values along with their indices
    orig_vals_inds = list(zip(get_orig_vals, loc1))
    # create a list of the indices for the nearest neighbours who fall within the threshold value
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2]))
    # get and create a list of the values associated 
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]])
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

output = snapshot_pattern(rand_array)
print(len(output))
