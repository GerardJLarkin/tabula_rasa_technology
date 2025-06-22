import warnings
warnings.filterwarnings("ignore")

# function to compare patterns for similarity
import numpy as np
import pandas as pd
from numba import njit, prange
import numba

## set similarity threshold limits
distance_threshold = 0.50
xc_perc = 0.50 
yc_perc = 0.50
xp_perc = 0.50
xn_perc = 0.50
yp_perc = 0.50
yn_perc = 0.50

# @njit(nopython=True, fastmath=True, parallel=True) #numba.float32(numba.types.Array(numba.float32, 2, "C"), numba.types.Array(numba.float32, 2, "C")), 
# this does appear to work
def pattern_compare_2d(new_patom, ref_array):
    # new patom: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr]
    # ref patoms: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr]
    # Extract the category column
    categories1 = new_patom[:, 5]
    categories2 = ref_array[:, 5]
    # Perform a Cartesian join
    # Expand dimensions to compare categories using broadcasting
    category_match = categories1[:, None] == categories2[None, :]
    # Get indices where categories match
    indices1, indices2 = np.where(category_match)
    # Select matching rows
    matched_array1 = new_patom[indices1]
    matched_array2 = ref_array[indices2]
    # Concatenate horizontally to form the final joined result
    cart = np.hstack((matched_array1, matched_array2))
    print(cart.shape)

    cart_x_diff = np.absolute(cart[:,1] - cart[:,9])
    cart_y_diff = np.absolute(cart[:,2] - cart[:,10])
    cart_quad_len = cart[:,6] * cart[:,14]
    cart_xc_d = np.absolute(cart[:,3] - cart[:,11])
    cart_yc_d = np.absolute(cart[:,4] - cart[:,12])
    cart_x_diff_sum = np.resize(cart_x_diff.sum(), cart_quad_len.shape[0])
    cart_y_diff_sum = np.resize(cart_y_diff.sum(), cart_quad_len.shape[0])
    cart_x_d = cart_x_diff_sum / cart_quad_len
    cart_y_d = cart_y_diff_sum / cart_quad_len
    cond = [(cart_xc_d <= 0.2) & (cart_yc_d <= 0.2) & (cart_x_d <= 0.3) & (cart_y_d <= 0.3)]
    choice = [1]
    cart_similar = np.select(cond, choice, 0)
    cart_arr = np.unique(np.column_stack((cart[:,:8], cart_xc_d, cart_yc_d, cart_x_d, cart_y_d, cart[:,15], cart_similar)), axis=0)
    
    return cart_arr