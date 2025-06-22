# function to compare patterns for similarity
import numpy as np
from os import listdir
import csv
from itertools import product
import sys
from time import perf_counter
sys.path.append('/home/gerard/Desktop/capstone_project')
from snapshot_3d_pattern_v6 import patoms3d
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings("ignore")

s = perf_counter()

np.random.seed(42)
rand_array = np.random.random((30, 720, 1280))
z_len = rand_array.shape[0]
y_len = rand_array.shape[1]
x_len = rand_array.shape[2]

distance_threshold = 0.05
colour_threshold = 0.2 # need to consider this more

# adding in comment line
def pattern_centroid_compare(arr1, arr2):
    nxc1 = arr1[-1:][0,4]; nxc2 = arr2[-1:][0,4]
    if nxc1 == nxc2:
        nxc_sim = 1/3
    else:
        nxc_sim = (1 - abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)) / 3
    nyc1 = arr1[-1:][0,5]; nyc2 = arr2[-1:][0,5] 
    if nyc1 == nyc2:
        nyc_sim = 1/3
    else:
        nyc_sim = (1 - abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)) / 3
    nzc1 = arr1[-1:][0,6]; nzc2 = arr2[-1:][0,6] 
    if nzc1 == nzc2:
        nzc_sim = 1/3
    else:
        nzc_sim = (1 - abs(nzc1 - nzc2)/((nzc1 + nzc2)/2)) / 3
    sim_perc = (nxc_sim + nyc_sim + nzc_sim)
    return sim_perc 

# adding in comment line
def distance_compare(arr1, arr2):
    # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
    dic = dict.fromkeys(list(range(arr1.shape[0] - 1)), 0)
    for (indx, i), j in product(enumerate(arr1[:-1,-2]), arr2[:-1,-2]):
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
    for (indx, i), j in product(enumerate(arr1[:-1,0]), arr2[:-1,0]):
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
        nxc_sim = 1/3
    else:
        nxc_sim = (1 - abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)) / 3
    nyc1 = arr1[-1:][0,2]; nyc2 = arr2[-1:][0,2] 
    if nyc1 == nyc2:
        nyc_sim = 1/3
    else:
        nyc_sim= (1 - abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)) / 3
    nzc1 = arr1[-1:][0,3]; nzc2 = arr2[-1:][0,3] 
    if nzc1 == nzc2:
        nzc_sim = 1/3
    else:
        nzc_sim = (1 - abs(nzc1 - nzc2)/((nzc1 + nzc2)/2)) / 3
    sim_perc = (nxc_sim + nyc_sim + nzc_sim)
    return sim_perc 

# adding in comment line
def pattern_compare_3d(list1, list2, ind):
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


# new_patoms = patoms3d(x_len, y_len, z_len, rand_array)

# compare_list = []
# atime = perf_counter()
# patom_indexes = list(product(range(len(new_patoms)), range(len(new_patoms))))
# with Pool(processes=cpu_count()) as pool:
#     items = [(new_patoms[i[0]], new_patoms[i[1]], ind) for ind, i in enumerate(patom_indexes)]
#     res = pool.starmap(pattern_compare_3d, items)
#     compare_list.append(res)
#     print("Time to compare 3D patterns with multiprocessing (secs):", (perf_counter()-atime)/60)

# print("Time to compare 3d patterns (mins):", (perf_counter()-s)/60)