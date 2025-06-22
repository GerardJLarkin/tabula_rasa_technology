# function to compare patterns for similarity
import numpy as np
from itertools import product
import sys
from time import perf_counter
sys.path.append('/home/gerard/Desktop/capstone_project')
from snapshot_2d_pattern_v1 import patoms2d
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore")

s = perf_counter()

np.random.seed(5555)
rand_array = np.random.random((30, 720, 1280))
z_len = rand_array.shape[0]
y_len = rand_array.shape[1]
x_len = rand_array.shape[2]

# similarity will be a measure of
# the length of the patterns - maybe?
# the percentage of pixels that fall within the distance threshold for the smaller pattern
# the percentage of pixels in the smaller pattern that fall within the distance threshold measured against the total number of pixels in the larger pattern
# the percentage difference in the pattern centroids
# the colour values of each pixel that are identified as falling within the threshold distance
# the orientation of the pattern - maybe?
# the positioning of the pattern in relation to other patterns
centroid_diff = 0.05
distance_threshold = 0.05
sim_threshold = 0.95
# a bit of thinking needs to in to this
colour_threshold = 0.20

# only want final row of pattern so consider extracting that before passing as arguments
def pattern_centroid_compare(pat1, pat2):
    nxc1 = pat1[-1:][0,4]; nxc2 = pat2[-1:][0,4]
    if nxc1 == nxc2:
        nxc_sim = 0.5
    else:
        nxc_sim = 1 - abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)
    nyc1 = pat1[-1:][0,5]; nyc2 = pat2[-1:][0,5] 
    if nyc1 == nyc2:
        nyc_sim = 0.5
    else:
        nyc_sim = 1 - abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)
    sim_perc = (nxc_sim + nyc_sim)
    return sim_perc 

# only intersted is final column of pattern so extract that before passing to argument
def distance_compare(pat1, pat2):
    # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
    dic = dict.fromkeys(list(range(pat1.shape[0] - 1)), 0)
    for (indx, i), j in product(enumerate(pat1[:-1,-1]), pat2[:-1,-1]):
        # check if pixel falls within the threshold of the distance difference
        if abs(i - j)/((i + j)/2) <= distance_threshold:
            # if within threshold increase the count of that normalised pixel centroid distance
            dic[indx] +=1
    # count the the number of normalised pixel locations who have values greater than 1
    sim_sum = sum(int(i) > 0 for i in dic.values())
    # calculated value to determine how similar (based on normalised pixel distiance from centroid) pattern A (smaller pattern) is to be pattern B (larger pattern)
    # do I need to consider if a single pixel location in pattern A is within the distance threshold for a large portion of pattern Bs norm pixel locations?
    sim_perc = sim_sum/(pat2.shape[0])
    return sim_perc

# adding in comment line
def pattern_compare(patom1, patom2, ind):
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


pairs_list = []
for frame in range(rand_array.shape[0]):
    frame_patoms = patoms2d(x_len, y_len, rand_array[frame,:,:])
    atime = perf_counter()
    patom_indexes = list(product(range(len(frame_patoms)), range(len(frame_patoms))))
    with Pool(processes=cpu_count()) as pool:
        items = [(frame_patoms[i[0]], frame_patoms[i[1]], ind) for ind, i in enumerate(patom_indexes)]
        res = pool.starmap(pattern_compare, items)
        print("Time to compare 2D patterns with multiprocessing (secs):", (perf_counter()-atime))


e = perf_counter()
print("Time to compare patterns (mins):", (e-s)/60)
