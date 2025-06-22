## add ignore warnings for now, will remove and debug once full algorithm is complete
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter
import numpy as np
import pandas as pd
import math
from operator import itemgetter
from multiprocessing import Pool, cpu_count
from itertools import product
import sys
import sqlite3

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

## call locally created functions
from snapshot_2d_pattern_v1 import patoms2d
from snapshot_3d_pattern_v6 import patoms3d
from pattern_2d_compare_v1 import pattern_compare

## start timer to asses how long process takes
s = perf_counter()

## create test data for algorithm development
np.random.seed(42)
rand_array = np.random.random((30, 720, 1280))
z_len = rand_array.shape[0]
y_len = rand_array.shape[1]
x_len = rand_array.shape[2]
    
compare_list = []
for frame in range(rand_array.shape[0]):
    frame_patoms = patoms2d(x_len, y_len, rand_array[frame,:,:])
    # compare all patoms against all other patoms in the frame, add to list that can hold patoms before comparing against exiting patoms
    atime = perf_counter()
    patom_indexes = list(product(range(len(frame_patoms)), range(len(frame_patoms))))
    with Pool(processes=cpu_count()) as pool:
        items = [(frame_patoms[i[0]], frame_patoms[i[1]], ind) for ind, i in enumerate(patom_indexes)]
        res = pool.starmap(pattern_compare, items)
        compare_list.append(res)
        #print("Time to compare 2D patterns with multiprocessing (secs):", (perf_counter()-atime))

# ingest data frame by frame
for frame in range(rand_array.shape[0]):
    #################################################################################
    ####################### FIRST TASK: FIND PATTERNS IN FRAME ######################
    #################################################################################
    similar_pattern_groups_list = []
    # find patterns in data
    frame_patoms = patoms2d(x_len, y_len, rand_array[frame,:,:])
    num_patoms = len(frame_patoms)
    #################################################################################
    ########## SECOND TASK: COMPARE ALL PATTERNS IN FRAME TO THEMSELVES #############
    #################################################################################
    ## compare all patoms against all other patoms in the frame, add to list that can hold patoms before comparing against exiting patoms
    atime = perf_counter()
    patom_indexes = list(product(range(num_patoms), range(num_patoms)))
    with Pool(processes=cpu_count()) as pool:
        items = [(frame_patoms[i[0]], frame_patoms[i[1]], ind) for ind, i in enumerate(patom_indexes)]
        ## function outputs ind value of the patom_indexes list, the centroid and distance similarity measures
        res = pool.starmap(pattern_compare_2d, items)
        ## create dictionary the holds as keys the index of each patom identified in the frame
        match_list = []
        ## loop through the output of the comparison function
        for ix, i in enumerate(res):
            ## pass if its the same patom in the frame being compared against itself
            if patom_indexes[ix][0] == patom_indexes[ix][1]:
                pass
            else:
                ## check if compared patterns fall within similarity threshold values
                if (i[1] >= centroid_sim_threshold) and (i[2] >= dist_sim_threshold):
                    match_list.append([patom_indexes[ix][0], patom_indexes[ix][1]])
                else:
                    pass
        # merge pattern indices that have similar elements
        similar_pattern_groups = merge_lists_with_common_elements(match_list)
        similar_pattern_groups_list.append(similar_pattern_groups)
        #print("Time to compare 2D patterns with multiprocessing (secs):", (perf_counter()-atime))
    ## add the frame index to the beginning of each row in each pattern
    table_frame_patoms = [[tuple([frame] + list(x)) for x in patoms] for patoms in frame_patoms]
    #############################################################################################################################################################
    ########## THIRD TASK: STORE NEW PATTERNS IN EMPTY TABLES AND ADD OLD (PATTERNS SIMILAR TO PREVIOUSLY RECEIVED DATA) IN THEIR RESPECTIVE TABLES #############
    #############################################################################################################################################################
    ## get all non-empty reference data tables from the database
    ref = cur2d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%ref%';")
    tables_ref = ref.fetchall()  # List of tuples with table names
    table_names_ref = []
    for (ref_name,) in tables_ref:  # Unpack the tuple
        cur2d.execute(f"SELECT COUNT(*) FROM {ref_name};")
        row_count = cur2d.fetchone()[0]  # Get the row count
        if row_count == 0:
            table_names_ref.append(ref_name)
        else:
            pass
    ## sort table names (in case they aren't)
    table_names_ref = sorted(table_names_ref)
    ## if there is at least one non-empty reference table then we compare current patterns against reference pattern
    if table_names_ref:
        for i in table_frame_patoms:
            for j in table_names_ref:
                if 

    ## get all non-empty non reference data tables from the database
    nonref = cur2d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%ref%';")
    tables_nonref = nonref.fetchall()  # List of tuples with table names
    table_names_nonref = []
    # Loop through each table and check if it's empty
    for (table_name,) in tables_nonref:  # Unpack the tuple
        cur2d.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cur2d.fetchone()[0]  # Get the row count
        ## perform comparison on tables that have data in the database
        if row_count == 0:
            table_names_nonref.append(table_name)
        else:
            pass
    ## sort table names (in case they aren't)
    table_names_nonref = sorted(table_names_nonref)
    
    ## insert data into empty tables
    for jx, j in enumerate(similar_pattern_groups):
        # now that I have the empty table I need to write all the patterns from each pattern group into it
        for patom in j:
            cur2d.executemany(f"INSERT INTO {table_names[jx]}(frame, colour, x_pos, y_pos, norm_x_pos, norm_y_pos, norm_dist, patom_ind) VALUES (?,?,?,?,?,?,?,?)", table_frame_patoms[patom])



## save 2D patterns to database and compare new patterns to existing patterns
# con2d = sqlite3.connect("database_2d.db")
# cur2d = con2d.cursor()


##################
#### PART ONE ####
##################

# np.random.seed(5555)
# rand_array = np.random.random((60, 720, 1280))

##################
#### PART TWO ####
##################

# z_len = rand_array.shape[0]
# y_len = rand_array.shape[1]
# x_len = rand_array.shape[2]

# threshold = 0.00005 #0.00005
# motion = [[-1, -1, -1], [0, -1, -1], [1, -1, -1], [-1, 0, -1], [0, 0, -1], [1, 0, -1], [-1, 1, -1], 
#         [0, 1, -1], [1, 1, -1], [-1, -1, 0], [0, -1, 0], [1, -1, 0], [-1, 0, 0], [1, 0, 0], [-1, 1, 0], 
#         [0, 1, 0], [1, 1, 0], [-1, -1, 1], [0, -1, 1], [1, -1, 1], [-1, 0, 1], [0, 0, 1], [1, 0, 1], 
#         [-1, 1, 1], [0, 1, 1], [1, 1, 1]]

# def snapshot_pattern(i):
#     orig_array = rand_array
    
#     indxs = motion[i]
#     if indxs[0] == 1:
#         ia = None; ib = -1; xa = 1; xb = None
#     if indxs[0] == 0:
#         ia = None; ib = None; xa = None; xb = None
#     if indxs[0] == -1:
#         ia = 1; ib = None; xa = None; xb = -1      
#     if indxs[1] == 1:
#         ja = None; jb = -1; ya = 1; yb = None
#     if indxs[1] == 0:
#         ja = None; jb = None; ya = None; yb = None
#     if indxs[1] == -1:
#         ja = 1; jb = None; ya = None; yb = -1
#     if indxs[2] == 1:
#         ka = None; kb = -1; za = 1; zb = None
#     if indxs[2] == 0:
#         ka = None; kb = None; za = None; zb = None
#     if indxs[2] == -1:
#         ka = 1; kb = None; za = None; zb = -1  

#     arr = orig_array[ia:ib, ja:jb, ka:kb]
#     comp =  orig_array[xa:xb, ya:yb, za:zb]
#     truth = abs(comp - arr) <= threshold # heavy not too bad ~27MiB
#     true_indices = np.asarray(truth).nonzero()
#     def get_orig_loc_i(x):
#         if indxs[0] == 1:
#             return x+1
#         if indxs[0] == 0:
#             return x
#         if indxs[0] == -1:
#             return x-1
#     def get_orig_loc_j(x):
#         if indxs[1] == 1:
#             return x+1
#         if indxs[1] == 0:
#             return x
#         if indxs[1] == -1:
#             return x-1
#     def get_orig_loc_k(x):
#         if indxs[2] == 1:
#             return x+1
#         if indxs[2] == 0:
#             return x
#         if indxs[2] == -1:
#             return x-1
#     orig_loc_i = np.apply_along_axis(get_orig_loc_i, 0, true_indices[0])
#     orig_loc_j = np.apply_along_axis(get_orig_loc_j, 0, true_indices[1])
#     orig_loc_k = np.apply_along_axis(get_orig_loc_k, 0, true_indices[2])
#     get_orig_vals = orig_array[orig_loc_i, orig_loc_j, orig_loc_k]
#     loc1 = list(zip(orig_loc_i/z_len, orig_loc_j/y_len, orig_loc_k/x_len))# 7.5 MiB
#     orig_vals_inds = list(zip(get_orig_vals, loc1))# 5 MiB
#     loc2 = list(zip(true_indices[0]/z_len, true_indices[1]/y_len, true_indices[2]/x_len)) # 7.5 MiB
#     get_tnn_vals = list(orig_array[true_indices[0], true_indices[1], true_indices[2]]) # 2.5 MiB
#     tnn_vals_inds = list(zip(get_tnn_vals, loc2))# 3.1 MiB
#     out = orig_vals_inds + tnn_vals_inds

#     return out

# def patoms():
#     # with multiprocessing
#     atime = perf_counter()
#     with Pool(processes=cpu_count()) as pool:
#         res = pool.map(snapshot_pattern, range(26))

#     # combine the outputs of each nearest neighbour function
#     combined_output = sorted(set([i for x in res for i in x]))

#     # split list when value between subsequent elements is greater than threshold
#     res, last = [[]], None
#     for x in combined_output:
#         if last is None or abs(last - x[0]) <= threshold: #runtime warning here
#             res[-1].append(x)
#         else:
#             res.append([x])
#         last = x[0]

#     # sort the lists of tuples based on the indices (need to get indices as tuple)
#     s_res = []
#     for i in res:
#         s = sorted(i, key=itemgetter(1))
#         if len(s) >= 10:
#             s_res.append(s)

#     # then need to obtain a normalised distance for all points from the 'center' of the pattern
#     norm_patoms = []
#     for pat in s_res:
#         pat_len = len(pat)
#         x = [p[1][0] for p in pat]
#         mean_x = sum(x)/pat_len; min_x = min(x); max_x = max(x)
#         norm_x = [(i - min_x)/(max_x - min_x) for i in x]
#         y = [p[1][1] for p in pat]
#         mean_y = sum(y)/pat_len; min_y = min(y); max_y = max(y)
#         norm_y = [(i - min_y)/(max_y - min_y) for i in y]
#         z = [p[1][2] for p in pat]
#         mean_z = sum(z)/pat_len; min_z = min(z); max_z = max(z)
#         norm_z = [(i - min_z)/(max_z - min_z) for i in z]
#         position_centroid = list((mean_x, mean_y, mean_z))
#         pattern_centroid = list((sum(norm_x)/pat_len, sum(norm_y)/pat_len, sum(norm_z)/pat_len))
#         centroid_list = [tuple([0.0]+position_centroid+pattern_centroid+[0.0])]
#         loc_norm = list(zip(norm_x, norm_y, norm_z))
#         norm_dist = list(map(lambda x: math.dist(pattern_centroid, list(x)), loc_norm))
#         val = [p[0] for p in pat]
#         patom = list(zip(val, x, y, z, norm_x, norm_y, norm_z, norm_dist))
#         patom = patom + centroid_list
#         norm_patoms.append(patom)
    
#     # with Pool(processes=cpu_count()) as pool:
#     #     norm_patoms = pool.map(pre_patom, s_res)

#     # structure each file with the pixel value, inde, etc. as a single row
#     # for ind, pat in enumerate(norm_patoms):
#     #     with open(f'/home/gerard/Desktop/capstone_project/patoms/pat{ind}.csv', 'w', newline='') as csvfile:
#     #         writer = csv.writer(csvfile, delimiter=',')
#     #         writer.writerows(pat)

#     print("Time to get patterns with multiprocessing (mins):", (perf_counter()-atime)/60)

#     return norm_patoms


####################
#### PART THREE ####
####################

# new_patoms = patoms()

# pair_comp = list(product(new_patoms, new_patoms))[:20]

# # similarity will be a measure of
# # the length of the patterns - maybe?
# # the percentage of pixels that fall within the distance threshold for the smaller pattern
# # the percentage of pixels in the smaller pattern that fall within the distance threshold measured against the total number of pixels in the larger pattern
# # the percentage difference in the pattern centroids
# # the colour values of each pixel that are identified as falling within the threshold distance
# # the orientation of the pattern - maybe?
# # the positioning of the pattern in relation to other patterns
# centroid_diff = 0.20
# distance_threshold = 0.08
# sim_threshold = 0.95
# # a bit of thinking needs to in to this
# colour_threshold = 0.20

# # adding in comment line
# def pattern_centroid_compare(arr1, arr2):
#     nxc1 = arr1[-1:][0,4]; nxc2 = arr2[-1:][0,4]
#     if nxc1 == nxc2:
#         nxc_diff = 0.0
#     else:
#         nxc_diff = abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)
#     nyc1 = arr1[-1:][0,5]; nyc2 = arr2[-1:][0,5] 
#     if nyc1 == nyc2:
#         nyc_diff = 0.0
#     else:
#         nyc_diff = abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)
#     nzc1 = arr1[-1:][0,6]; nzc2 = arr2[-1:][0,6] 
#     if nzc1 == nzc2:
#         nzc_diff = 0.0
#     else:
#         nzc_diff = abs(nzc1 - nzc2)/((nzc1 + nzc2)/2)
#     sim_perc = (nxc_diff + nyc_diff + nzc_diff)
#     return sim_perc 

# # adding in comment line
# def distance_compare(arr1, arr2):
#     # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
#     dic = dict.fromkeys(list(range(arr1.shape[0] - 1)), 0)
#     for (indx, i), j in product(enumerate(arr1[:,-1]), arr2[:,-1]):
#         # check if pixel falls within the threshold of the distance difference
#         if abs(i - j)/((i + j)/2) <= distance_threshold:
#             # if within threshold increase the count of that normalised pixel centroid distance
#             dic[indx] +=1
#     # count the the number of normalised pixel locations who have values greater than 1
#     sim_sum = sum(int(i) > 0 for i in dic.values())
#     # calculated value to determine how similar (based on normalised pixel distiance from centroid) pattern A (smaller pattern) is to be pattern B (larger pattern)
#     # do I need to consider if a single pixel location in pattern A is within the distance threshold for a large portion of pattern Bs norm pixel locations?
#     sim_perc = sim_sum/(arr2.shape[0] - 1)
#     return sim_perc 

# # adding in comment line    
# def colour_compare(arr1, arr2):
#     # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
#     dic = dict.fromkeys(list(range(arr1.shape[0] - 1)), 0)
#     for (indx, i), j in product(enumerate(arr1[:,0]), arr2[:,0]):
#         # check if pixel falls within the threshold of the colour difference
#         if abs(i - j)/((i + j)/2) <= colour_threshold:
#             # if within threshold increase the count of that pixel colour
#             dic[indx] +=1
#     # count the the number of pixel colours who have values greater than 1
#     sim_sum = sum(int(i) > 0 for i in dic.values())
#     sim_perc = sim_sum/(arr2.shape[0] - 1)
#     return sim_perc 

# # adding in comment line   
# def position_centroid_compare(arr1, arr2):
#     nxc1 = arr1[-1:][0,1]; nxc2 = arr2[-1:][0,1]
#     if nxc1 == nxc2:
#         nxc_diff = 0.0
#     else:
#         nxc_diff = abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)
#     nyc1 = arr1[-1:][0,2]; nyc2 = arr2[-1:][0,2] 
#     if nyc1 == nyc2:
#         nyc_diff = 0.0
#     else:
#         nyc_diff = abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)
#     nzc1 = arr1[-1:][0,3]; nzc2 = arr2[-1:][0,3] 
#     if nzc1 == nzc2:
#         nzc_diff = 0.0
#     else:
#         nzc_diff = abs(nzc1 - nzc2)/((nzc1 + nzc2)/2)
#     sim_perc = (nxc_diff + nyc_diff + nzc_diff)
#     return sim_perc 

# # adding in comment line
# def pattern_compare(ind, list1, list2):
#     data_array1 = np.array(list1, dtype=float)
#     shape1 = data_array1.shape[0]
#     data_array2 = np.array(list2, dtype=float)
#     shape2 = data_array2.shape[0]
#     if shape1 <= shape2:
#         centroid = pattern_centroid_compare(data_array1, data_array2)
#         dist = distance_compare(data_array1, data_array2)
#         colour = colour_compare(data_array1, data_array2)
#         position = position_centroid_compare(data_array1, data_array2)
#     else:
#         centroid = pattern_centroid_compare(data_array2, data_array1)
#         dist = distance_compare(data_array2, data_array1)
#         colour = colour_compare(data_array2, data_array1)
#         position = position_centroid_compare(data_array2, data_array1)

#     return [ind, centroid, dist, colour, position]

# # adding in comment line
# def run_compare(ind):
#     data = pair_comp[ind]
#     output = pattern_compare(ind, data[0], data[1])
    
#     return output

# with Pool(processes=cpu_count()) as pool:
#     res = pool.map(run_compare, range(len(pair_comp)))
#     compared = list(zip(pair_comp, res))

# e = perf_counter()
# print("Time to compare patterns (mins):", (e-s)/60)
