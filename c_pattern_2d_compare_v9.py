import warnings
warnings.filterwarnings("ignore")
import numpy as np

def pattern_compare_2d(new_patom, ref_array):
    # new patom: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, segment, segment_cnt, frame_ind_arr] + [col_d, xc_d, yc_d, x_d, y_d] + [segment_similar]
    # ref patoms: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, segment, segment_cnt, frame_ind_arr]

    # Cartesian row join
    np_repeat = np.repeat(new_patom, ref_array.shape[0], axis=0)  # Repeat rows of A
    ref_array_tile = np.tile(ref_array, (new_patom.shape[0], 1))  # Tile rows of B

    # Merge using hstack (concatenation along columns)
    cart = np.hstack((np_repeat, ref_array_tile))
    mask = cart[:,5] == cart[:,13]
    cart = cart[mask]
    cart_col_diff = np.absolute(cart[:,0] - cart[:,8])
    cart_x_diff = np.absolute(cart[:,1] - cart[:,9])
    cart_y_diff = np.absolute(cart[:,2] - cart[:,10])
    cart_segment_len = cart[:,6] * cart[:,14]
    cart_xc_d = np.absolute(cart[:,3] - cart[:,11])
    cart_yc_d = np.absolute(cart[:,4] - cart[:,12])
    cart_x_diff_sum = np.resize(cart_x_diff.sum(), cart_segment_len.shape[0])
    cart_y_diff_sum = np.resize(cart_y_diff.sum(), cart_segment_len.shape[0])
    cart_x_d = cart_x_diff_sum / cart_segment_len
    cart_y_d = cart_y_diff_sum / cart_segment_len
    cart_col_diff_sum = np.resize(cart_col_diff.sum(), cart_segment_len.shape[0])
    cart_c_d = cart_col_diff_sum / cart_segment_len
    cond = [(cart_c_d <= 0.2) &(cart_xc_d <= 0.2) & (cart_yc_d <= 0.2) & (cart_x_d <= 0.3) & (cart_y_d <= 0.3)]
    choice = [1]
    cart_similar = np.select(cond, choice, 0)
    cart_arr = np.unique(np.column_stack((cart[:,:8], cart_col_diff, cart_xc_d, cart_yc_d, cart_x_d, cart_y_d, cart[:,15], cart_similar)), axis=0)
    
    return cart_arr