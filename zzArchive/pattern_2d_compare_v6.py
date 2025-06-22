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

# @njit(numba.float64(numba.types.Array(numba.float64, 2, "C")), nopython=True, fastmath=True, parallel=True)
# this does appear to work
def pattern_compare_2d(new_patom, ref_arrays):
    # new patom: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr]
    # ref patoms: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, frame_ind_arr]
    # Extract the category column
    categories1 = new_patom[:, 5]
    categories2 = ref_arrays[:, 5]
    # Perform a Cartesian join
    #m, n = len(new_patom), len(ref_arrays)
    # Expand dimensions to compare categories using broadcasting
    category_match = categories1[:, None] == categories2[None, :]
    # Get indices where categories match
    indices1, indices2 = np.where(category_match)
    # Select matching rows
    matched_array1 = new_patom[indices1]
    matched_array2 = ref_arrays[indices2]
    # Concatenate horizontally to form the final joined result
    cart = np.hstack((matched_array1, matched_array2))

    cart_df = pd.DataFrame(cart, columns=['pcol','px','py','pxc','pyc','pq','pqlen','pfind','rcol','rx','ry','rxc','ryc','rq','rqlen','rfind'])
    cart_df['x_diff'] = abs(cart_df.iloc[:,1] - cart_df.iloc[:,9])
    cart_df['y_diff'] = abs(cart_df.iloc[:,2] - cart_df.iloc[:,10])
    cart_df['quad_len'] = cart_df.iloc[:,6] * cart_df.iloc[:,14]
    cart_df['xc_d'] = abs(cart_df.iloc[:,3] - cart_df.iloc[:,11])
    cart_df['yc_d'] = abs(cart_df.iloc[:,4] - cart_df.iloc[:,12])
    cart_df = cart_df.groupby(['pcol','px','py','pxc','pyc','pq','pqlen','pfind','rcol','rfind','xc_d','yc_d']).agg({'x_diff': 'sum', 'y_diff': 'sum', 'quad_len': 'min'})
    cart_df['x_d'] = cart_df['x_diff'] / cart_df['quad_len']
    cart_df['y_d'] = cart_df['y_diff'] / cart_df['quad_len']
    cart_df = cart_df.reset_index()[['pcol','px','py','pxc','pyc','pq','pqlen','pfind','rcol','rfind','xc_d','yc_d','x_d','y_d']] 
    cond = [(cart_df['xc_d'] <= 0.2) & (cart_df['yc_d'] <= 0.2) & (cart_df['x_d'] <= 0.3) & (cart_df['y_d'] <= 0.3)]
    choice = [1]
    cart_df['similar'] = np.select(cond, choice, 0)
    cart_df = cart_df[['pcol','px','py','pxc','pyc','pq','pqlen','pfind','xc_d','yc_d','x_d','y_d','rfind','similar']].to_numpy()
    
    return cart_df

# @njit(parallel=True)
# def pattern_compare_2d(a, b):
#     rows_a, cols_a = a.shape
#     rows_b, cols_b = b.shape
#     # Preallocate the result array
#     result = np.empty((rows_a * rows_b, cols_a + cols_b), dtype=a.dtype)

#     # Perform the Cartesian product
#     for i in prange(rows_a):
#         for j in range(rows_b):
#             # Combine row i from `a` with row j from `b`
#             result[i * rows_b + j, :cols_a] = a[i]
#             result[i * rows_b + j, cols_a:] = b[j]

#     print(result.nbytes)