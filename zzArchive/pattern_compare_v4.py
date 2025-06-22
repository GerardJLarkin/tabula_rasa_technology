# function to compare patterns for similarity
import numpy as np
from os import listdir
import csv
from itertools import product
import sys
from time import perf_counter
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
from strt_vid_mpt import patoms
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore")

s = perf_counter()

new_patoms = patoms()

pair_comp = list(product(new_patoms, new_patoms))

# similarity will be a measure of
# the length of the patterns - maybe?
# the percentage of pixels that fall within the distance threshold for the smaller pattern
# the percentage of pixels in the smaller pattern that fall within the distance threshold measured against the total number of pixels in the larger pattern
# the percentage difference in the pattern centroids
# the colour values of each pixel that are identified as falling within the threshold distance
# the orientation of the pattern - maybe?
# the positioning of the pattern in relation to other patterns
centroid_diff = 0.20
distance_threshold = 0.08
sim_threshold = 0.95
# a bit of thinking needs to in to this
colour_threshold = 0.20

# adding in comment line
def pattern_centroid_compare(arr1, arr2):
    nxc1 = arr1[-1:][0,4]; nxc2 = arr2[-1:][0,4]
    if nxc1 == nxc2:
        nxc_diff = 0.0
    else:
        nxc_diff = abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)
    nyc1 = arr1[-1:][0,5]; nyc2 = arr2[-1:][0,5] 
    if nyc1 == nyc2:
        nyc_diff = 0.0
    else:
        nyc_diff = abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)
    nzc1 = arr1[-1:][0,6]; nzc2 = arr2[-1:][0,6] 
    if nzc1 == nzc2:
        nzc_diff = 0.0
    else:
        nzc_diff = abs(nzc1 - nzc2)/((nzc1 + nzc2)/2)
    sim_perc = (nxc_diff + nyc_diff + nzc_diff)
    return sim_perc 

# adding in comment line
def distance_compare(arr1, arr2):
    # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
    dic = dict.fromkeys(list(range(arr1.shape[0] - 1)), 0)
    for (indx, i), j in product(enumerate(arr1[:,-1]), arr2[:,-1]):
        # check if pixel falls within the threshold of the distance difference
        if abs(i - j)/((i + j)/2) <= distance_threshold:
            # if within threshold increase the count of that normalised pixel centroid distance
            dic[indx] +=1
    # count the the number of normalised pixel locations who have values greater than 1
    sim_sum = sum(int(i) > 0 for i in dic.values())
    # calculated value to determine how similar (based on normalised pixel distiance from centroid) pattern A (smaller pattern) is to be pattern B (larger pattern)
    # do I need to consider if a single pixel location in pattern A is within the distance threshold for a large portion of pattern Bs norm pixel locations?
    sim_perc = sim_sum/(arr2.shape[0] - 1)
    return sim_perc 

# adding in comment line    
def colour_compare(arr1, arr2):
    # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
    dic = dict.fromkeys(list(range(arr1.shape[0] - 1)), 0)
    for (indx, i), j in product(enumerate(arr1[:,0]), arr2[:,0]):
        # check if pixel falls within the threshold of the colour difference
        if abs(i - j)/((i + j)/2) <= colour_threshold:
            # if within threshold increase the count of that pixel colour
            dic[indx] +=1
    # count the the number of pixel colours who have values greater than 1
    sim_sum = sum(int(i) > 0 for i in dic.values())
    sim_perc = sim_sum/(arr2.shape[0] - 1)
    return sim_perc 

# adding in comment line   
def position_centroid_compare(arr1, arr2):
    nxc1 = arr1[-1:][0,1]; nxc2 = arr2[-1:][0,1]
    if nxc1 == nxc2:
        nxc_diff = 0.0
    else:
        nxc_diff = abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)
    nyc1 = arr1[-1:][0,2]; nyc2 = arr2[-1:][0,2] 
    if nyc1 == nyc2:
        nyc_diff = 0.0
    else:
        nyc_diff = abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)
    nzc1 = arr1[-1:][0,3]; nzc2 = arr2[-1:][0,3] 
    if nzc1 == nzc2:
        nzc_diff = 0.0
    else:
        nzc_diff = abs(nzc1 - nzc2)/((nzc1 + nzc2)/2)
    sim_perc = (nxc_diff + nyc_diff + nzc_diff)
    return sim_perc 

# adding in comment line
def pattern_compare(ind, list1, list2):
    data_array1 = np.array(list1, dtype=float)
    shape1 = data_array1.shape[0]
    data_array2 = np.array(list2, dtype=float)
    shape2 = data_array2.shape[0]
    if shape1 <= shape2:
        centroid = pattern_centroid_compare(data_array1, data_array2)
        dist = distance_compare(data_array1, data_array2)
        colour = colour_compare(data_array1, data_array2)
        position = position_centroid_compare(data_array1, data_array2)
    else:
        centroid = pattern_centroid_compare(data_array2, data_array1)
        dist = distance_compare(data_array2, data_array1)
        colour = colour_compare(data_array2, data_array1)
        position = position_centroid_compare(data_array2, data_array1)

    return [ind, centroid, dist, colour, position]

# adding in comment line
def run_compare(ind):
    data = pair_comp[ind]
    output = pattern_compare(ind, data[0], data[1])
    
    return output

with Pool(processes=cpu_count()) as pool:
    res = pool.map(run_compare, range(len(pair_comp)))
    compared = list(zip(pair_comp, res))

e = perf_counter()
print("Time to compare patterns (mins):", (e-s)/60)
