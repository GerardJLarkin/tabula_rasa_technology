# https://stackoverflow.com/questions/70172677/compare-value-in-a-2d-orig_array-to-nearby-values
import numpy as np
from operator import itemgetter
import math
from time import perf_counter

np.random.seed(5555)
rand_array = np.random.random((30, 720, 1280))

motion = np.transpose(np.indices((3,3,3)) - 1).reshape(-1, 3)
motion = np.delete(motion, 13, axis=0)
print(motion.tolist())

# def snapshot_pattern(array):
#     strt_stack = perf_counter()
#     threshold = 0.00005
#     orig_array = array
    
#     n1_arr = orig_array[1:,1:,1:]
#     n1_comp = orig_array[:-1,:-1,:-1]
#     #[-1, -1, -1]

#     n2_arr = orig_array[1:,1:,:]
#     n2_comp = orig_array[:-1,:-1,:]
#     #[-1, -1,  0]
    
#     n3_arr = orig_array[1:,1:,:-1]
#     n3_comp = orig_array[:-1,:-1,1:]
#     #[-1, -1,  1] 
    
#     n4_arr = orig_array[1:,:,1:]
#     n4_comp = orig_array[:-1,:,:-1]
#     #[-1,  0, -1] 

#     n5_arr = orig_array[1:,:,:]
#     n5_comp = orig_array[:-1,:,:]
#     #[-1,  0,  0] 
     
#     n6_arr = orig_array[1:,:,:-1]
#     n6_comp = orig_array[:-1,:,1:]
#     #[-1,  0,  1] 

#     n7_arr = orig_array[1:,:-1,1:] 
#     n7_comp = orig_array[:-1,1:,:-1]
#     #[-1,  1, -1] 
    
#     n8_arr = orig_array[1:,:-1,:]
#     n8_comp = orig_array[:-1,1:,:]
#     #[-1,  1,  0] 
    
#     n9_arr = orig_array[1:,:-1,:-1]
#     n9_comp = orig_array[:-1,1:,1:]
#     #[-1,  1,  1] 

#     n10_arr = orig_array[:,1:,1:]
#     n10_comp = orig_array[:,:-1,:-1]
#     #[ 0, -1, -1]
    
#     n11_arr = orig_array[:,1:,:] 
#     n11_comp = orig_array[:,:-1,:] 
#     #[ 0, -1,  0] 

#     n12_arr = orig_array[:,1:,:-1] 
#     n12_comp = orig_array[:,:-1,1:] 
#     #[ 0, -1,  1]     
    
#     n13_arr = orig_array[:,:,1:] 
#     n13_comp = orig_array[:,:,:-1] 
#     #[ 0,  0, -1]
    
#     n14_arr = orig_array[:,:,:-1] 
#     n14_comp = orig_array[:,:,1:] 
#     #[ 0,  0,  1]
    
#     n15_arr = orig_array[:,:-1,1:] 
#     n15_comp = orig_array[:,1:,:-1]
#     #[ 0,  1, -1]
    
#     n16_arr = orig_array[:,:-1,:] 
#     n16_comp = orig_array[:,1:,:]
#     #[ 0,  1,  0]
    
#     n17_arr = orig_array[:,:-1,:-1] 
#     n17_comp = orig_array[:,1:,1:] 
#     #[ 0,  1,  1]

#     n18_arr = orig_array[:-1,1:,1:] 
#     n18_comp = orig_array[1:,:-1,:-1] 
#     #[ 1, -1, -1]
    
#     n19_arr = orig_array[:-1,1:,:] 
#     n19_comp = orig_array[1:,:-1,:] 
#     #[ 1, -1,  0]
    
#     n20_arr = orig_array[:-1,1:,:-1] 
#     n20_comp = orig_array[1:,:-1,1:] 
#     #[ 1, -1,  1]
    
#     n21_arr = orig_array[:-1,:,1:]
#     n21_comp = orig_array[1:,:,:-1]
#     #[ 1,  0, -1]
    
#     n22_arr = orig_array[:-1,:,:] 
#     n22_comp = orig_array[1:,:,:] 
#     #[ 1,  0,  0]
    
#     n23_arr = orig_array[:-1,:,:-1]
#     n23_comp = orig_array[1:,:,1:]
#     #[ 1,  0,  1]  
    
#     n24_arr = orig_array[:-1,:-1,1:]  
#     n24_comp = orig_array[1:,1:,:-1] 
#     #[ 1,  1, -1] 
    
#     n25_arr = orig_array[:-1,:-1,:]
#     n25_comp = orig_array[1:,1:,:]
#     #[ 1,  1,  0]
    
#     n26_arr = orig_array[:-1,:-1,:-1] 
#     n26_comp = orig_array[1:,1:,1:] 
#     #[ 1,  1,  1] 
    
#     slice_arrs = [n1_arr, n2_arr, n3_arr, n4_arr, n5_arr, n6_arr, n7_arr, n8_arr, n9_arr, n10_arr,
#                   n11_arr, n12_arr, n13_arr, n14_arr, n15_arr, n16_arr, n17_arr, n18_arr,
#                   n19_arr, n20_arr, n21_arr, n22_arr, n23_arr, n24_arr, n25_arr, n26_arr]
#     comp_arrs = [n1_comp, n2_comp, n3_comp, n4_comp, n5_comp, n6_comp, n7_comp, n8_comp, n9_comp, n10_comp,
#                   n11_comp, n12_comp, n13_comp, n14_comp, n15_comp, n16_comp, n17_comp, n18_comp,
#                   n19_comp, n20_comp, n21_comp, n22_comp, n23_comp, n24_comp, n25_comp, n26_comp]
    
#     indices_pos = [[-1, -1, -1], [-1, -1,  0], [-1, -1,  1], [-1,  0, -1], [-1,  0,  0], [-1,  0,  1], [-1,  1, -1], [-1,  1,  0] , [-1,  1,  1],  
#     [0, -1, -1], [ 0, -1,  0], [ 0, -1,  1], [ 0,  0, -1], [ 0,  0,  1], [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1], [ 1, -1, -1], [ 1, -1,  0], 
#     [ 1, -1,  1], [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1], [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1]]
    
#     outs = []
#     for i in range(26):
#         truth = abs(comp_arrs[i] - slice_arrs[i]) <= threshold # heavy not too bad ~27MiB
#         true_indices = np.asarray(truth).nonzero()
#         def get_orig_loc_i(x):
#             if indices_pos[i][0] == 1:
#                 return x+1
#             if indices_pos[i][0] == 0:
#                 return x
#             if indices_pos[i][0] == -1:
#                 return x-1
#         def get_orig_loc_j(x):
#             if indices_pos[i][1] == 1:
#                 return x+1
#             if indices_pos[i][1] == 0:
#                 return x
#             if indices_pos[i][1] == -1:
#                 return x-1
#         def get_orig_loc_k(x):
#             if indices_pos[i][2] == 1:
#                 return x+1
#             if indices_pos[i][2] == 0:
#                 return x
#             if indices_pos[i][2] == -1:
#                 return x-1
#         orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
#         orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
#         orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
#         get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
#         loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))# 7.5 MiB
#         orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
#         loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2])) # 7.5 MiB
#         get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
#         tnn_vals_inds = list(zip(get_tnn_vals, loc2))# 3.1 MiB
#         out = orig_vals_inds + tnn_vals_inds
#         outs.append(out)
    
#     # combine the outputs of each nearest neighbour function
#     combined_output = sorted(set([i for x in outs for i in x]))
    
#     # split list when value between subsequent elements is greater than threshold
#     res, last = [[]], None
#     for x in combined_output:
#         if last is None or abs(last - x[0]) <= threshold: #runtime warning here
#             res[-1].append(x)
#         else:
#             res.append([x])
#         last = x[0]

#     # sort the lists of tuples based on the indices (need to get indices as tuple)
#     s_res = []
#     for i in res:
#         s = sorted(i, key=itemgetter(1))
#         if len(s) >= 10:
#             s_res.append(s)

#     # then need to obtain a normalised distance for all points from the 'center' of the pattern
#     norm_patoms = []
#     for ind, pat in enumerate(s_res):
#         x = [p[1][0] for p in pat]
#         min_x = min(x); max_x = max(x)
#         norm_x = [(i - min_x)/(max_x - min_x) for i in x]
#         y = [p[1][1] for p in pat]
#         min_y = min(y); max_y = max(y)
#         norm_y = [(i - min_y)/(max_y - min_y) for i in y]
#         z = [p[1][2] for p in pat]
#         min_z = min(z); max_z = max(z)
#         norm_z = [(i - min_z)/(max_z - min_z) for i in z]
#         centroid = list((sum(x) / len(pat), sum(y) / len(pat), sum(z) / len(pat)))
#         centroid_norm = list((sum(norm_x) / len(pat), sum(norm_y) / len(pat), sum(norm_z) / len(pat)))
#         centroid_list = [tuple([0.0]+centroid+centroid_norm+[0.0])]
#         loc = [p[1] for p in pat]
#         dist = list(map(lambda x: math.dist(centroid, list(x)), loc))
#         tot = sum(dist)
#         # normalised distance value is from centroid
#         norm_dist = [i/tot for i in dist]
#         val = [p[0] for p in pat]
#         patom = list(zip(val, x, y, z, norm_x, norm_y, norm_z, norm_dist))
#         patom = patom + centroid_list
#         norm_patoms.append(patom)

#     end_stack = perf_counter()

#     stacked_time = end_stack - strt_stack
#     print('Took this many seconds: ', stacked_time)

#     return norm_patoms

# output = snapshot_pattern(rand_array)
# print(len(output))
