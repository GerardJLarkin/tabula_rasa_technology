# unnormalise function
import numpy as np

def unnormalise_xy(arr):

    out = arr.copy()

    # mean/mins/maxes from rows 0 and 1
    mean_x, mean_y = out[0, 1], out[0, 2]
    min_x, max_x, min_y, max_y = out[1, 0], out[1, 1], out[1, 2], out[1, 3]

    # normalised coords 
    norm_xy = out[2:, :2]
    
    ## need to add 2 rows on top of norm_xy
    fake_rows = np.array([[np.nan, np.nan],
                          [np.nan, np.nan]])
    
    norm_xy_2 = np.vstack((fake_rows, norm_xy))
    
    # unnormalise x coordinate
    r_x = max(max_x - mean_x, mean_x - min_x) 
    if r_x == 0:
        # in the normalisation step in the patom function 0 is inserted as the default value when the radius is zero
        # when being recovered the centroid x value is inserted in its place
        x_rec = np.full_like(norm_xy[:, 0], float(mean_x), dtype='float32')
    else:
        # recoovered x coordinate
        x_rec = mean_x + norm_xy[:, 0] * r_x

    r_y = max(max_y - mean_y, mean_y - min_y) 
    if r_y == 0:
        y_rec = np.full_like(norm_xy[:, 1], float(mean_y), dtype='float32')
    else:
        y_rec = mean_y + norm_xy[:, 1] * r_y

    x_rec = np.rint(x_rec).astype(np.int64)
    y_rec = np.rint(y_rec).astype(np.int64)

    out[2:, 0] = x_rec
    out[2:, 1] = y_rec  
    
    out = np.hstack((out, norm_xy_2))
    
    return out