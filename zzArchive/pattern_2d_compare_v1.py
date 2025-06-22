import warnings
warnings.filterwarnings("ignore")

# function to compare patterns for similarity
import numpy as np
from itertools import product

distance_threshold = 0.05

# only want final row of pattern so consider extracting that before passing as arguments
def pattern_centroid_compare(pat1, pat2):
    nxc1 = pat1[-1:][0,3]; nxc2 = pat2[-1:][0,3]
    if nxc1 == nxc2:
        nxc_sim = 0.5
    else:
        nxc_sim = (1 - abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)) / 2
    nyc1 = pat1[-1:][0,4]; nyc2 = pat2[-1:][0,4] 
    if nyc1 == nyc2:
        nyc_sim = 0.5
    else:
        nyc_sim = (1 - abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)) / 2
    sim_perc = (nxc_sim + nyc_sim)
    return sim_perc 

# only intersted is final column of pattern so extract that before passing to argument
def distance_compare(pat1, pat2):
    # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
    dic = dict.fromkeys(list(range(pat1.shape[0] - 1)), 0)
    for (indx, i), j in product(enumerate(pat1[:-1,-2]), pat2[:-1,-2]):
        # check if pixel falls within the threshold of the distance difference
        if abs(i - j)/((i + j)/2) <= distance_threshold:
            # if within threshold increase the count of that normalised pixel centroid distance
            dic[indx] +=1
    # count the the number of normalised pixel locations who have values greater than 1
    sim_sum = sum(int(i) > 0 for i in dic.values())
    # calculated value to determine how similar (based on normalised pixel distiance from centroid) pattern A (smaller pattern) is to be pattern B (larger pattern)
    # do I need to consider if a single pixel location in pattern A is within the distance threshold for a large portion of pattern Bs norm pixel locations?
    sim_perc = sim_sum/(pat2.shape[0] - 1)
    return sim_perc

# adding in comment line
def pattern_compare_2d(patom1, patom2, ind):
    data_patom1 = np.array(patom1, dtype=float)
    shape1 = data_patom1.shape[0]
    data_patom2 = np.array(patom2, dtype=float)
    shape2 = data_patom2.shape[0]
    if shape1 <= shape2:
        centroid = pattern_centroid_compare(data_patom1, data_patom2)
        dist = distance_compare(data_patom1, data_patom2)
    else:
        centroid = pattern_centroid_compare(data_patom2, data_patom1)
        dist = distance_compare(data_patom2, data_patom1)
     
    # returns the index of the patom pair comparison along with the similarity values
    return [ind, centroid, dist]