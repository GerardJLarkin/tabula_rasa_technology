# https://stackoverflow.com/questions/70172677/compare-value-in-a-2d-orig_array-to-nearby-values
import numpy as np

np.random.seed(5555)
rand_array = np.random.random((30, 720, 1280))
# print(rand_array.nbytes)

# time_array = np.load('/home/gerard/Desktop/capstone_project/norm_file.npy')
# print(time_array.nbytes)

motion = [[-1, -1, -1], [0, -1, -1], [1, -1, -1], [-1, 0, -1], [0, 0, -1], [1, 0, -1], [-1, 1, -1], 
          [0, 1, -1], [1, 1, -1], [-1, -1, 0], [0, -1, 0], [1, -1, 0], [-1, 0, 0], [1, 0, 0], [-1, 1, 0], 
          [0, 1, 0], [1, 1, 0], [-1, -1, 1], [0, -1, 1], [1, -1, 1], [-1, 0, 1], [0, 0, 1], [1, 0, 1], 
          [-1, 1, 1], [0, 1, 1], [1, 1, 1]]

def snapshot_pattern(array, i):
    threshold = 0.00005
    orig_array = array
    
    indxs = motion[i]
    if indxs[0] == 1:
        ia = None; ib = -1
        xa = 1; xb = None
    if indxs[0] == -1:
        ia = 1; ib = None
        xa = None; xb = -1      
    if indxs[1] == 1:
        ja = None; jb = -1
        ya = 1; yb = None
    if indxs[1] == -1:
        ja = 1; jb = None
        ya = None; yb = -1
    if indxs[2] == 1:
        ka = None; kb = -1
        za = 1; zb = None
    if indxs[2] == -1:
        ka = 1; kb = None
        za = None; zb = -1  

    arr = orig_array[ia:ib, ja:jb, ka:kb]
    comp =  orig_array[xa:xb, ya:yb, za:zb]
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
    def get_orig_loc_k(x):
        if indxs[2] == 1:
            return x+1
        if indxs[2] == 0:
            return x
        if indxs[2] == -1:
            return x-1
    orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
    orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
    orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
    get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
    loc1 = list(zip(orig_loc_i, orig_loc_j, orig_loc_k))# 7.5 MiB
    orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
    loc2 = list(zip(true_indices[0], true_indices[1], true_indices[2])) # 7.5 MiB
    get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
    tnn_vals_inds = list(zip(get_tnn_vals, loc2))# 3.1 MiB
    out = orig_vals_inds + tnn_vals_inds
    
    return out
