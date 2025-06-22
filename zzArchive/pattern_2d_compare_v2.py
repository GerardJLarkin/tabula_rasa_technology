import warnings
warnings.filterwarnings("ignore")

# function to compare patterns for similarity
import numpy as np
from itertools import product

distance_threshold = 0.05

# only want final row of pattern so consider extracting that before passing as arguments
def pattern_centroid_compare(pat_cent_x1, pat_cent_y1, pat_cent_x2, pat_cent_y2):
    if pat_cent_x1 == pat_cent_x2:
        nxc_sim = 0.5
    else:
        nxc_sim = (1 - abs(pat_cent_x1 - pat_cent_x2)/((pat_cent_x1 + pat_cent_x2)/2)) / 2
    if pat_cent_y1 == pat_cent_y2:
        nyc_sim = 0.5
    else:
        nyc_sim = (1 - abs(pat_cent_y1 - pat_cent_y2)/((pat_cent_y1 + pat_cent_y2)/2)) / 2
    sim_perc = (nxc_sim + nyc_sim)
    return sim_perc 

# only intersted is final column of pattern so extract that before passing to argument
def distance_compare(dist1_x, dist1_y, dist2_x, dist2_y):
    arr1_x = np.array(dist1_x)
    arr1_y = np.array(dist1_y)
    arr2_x = np.array(dist2_x)
    arr2_y = np.array(dist2_y)

    diffx = np.abs(arr1_x[:, None] - arr2_x)
    maxx = np.maximum(np.abs(arr1_x[:, None]), np.abs(arr2_x))
    sim_perc_x = (diffx / maxx).flatten()
    sim_perc_x = len(sim_perc_x[sim_perc_x <= distance_threshold].tolist())

    diffy = np.abs(arr1_y[:, None] - arr2_y)
    maxy = np.maximum(np.abs(arr1_y[:, None]), np.abs(arr2_y))
    sim_perc_y = (diffy / maxy).flatten()
    sim_perc_y = len(sim_perc_y[sim_perc_y <= distance_threshold].tolist())

    total_lists_length = len(dist1_x) * len(dist2_x)
    sim_perc_x = sim_perc_x/total_lists_length
    sim_perc_y = sim_perc_y/total_lists_length
    
    return sim_perc_x, sim_perc_y

# adding in comment line
def pattern_compare_2d(pat1_dist, pat1_cent, pat2_dist, pat2_cent, i):
    centroid_sim = pattern_centroid_compare(pat1_cent[0], pat1_cent[1], pat2_cent[0], pat2_cent[1])
    dist_sim = distance_compare(pat1_dist[0], pat1_dist[1], pat2_dist[0], pat2_dist[1])
     
    # returns the index of the patom pair comparison along with the similarity values
    return [centroid_sim, dist_sim, i]