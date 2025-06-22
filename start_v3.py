## add ignore warnings for now, will remove and debug once full algorithm is complete
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter
import numpy as np
#import pandas as pd
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
from snapshot_2d_pattern_v3 import patoms2d
from snapshot_3d_pattern_v6 import patoms3d
from pattern_2d_compare_v3 import pattern_compare_2d
from pattern_3d_compare_v4 import pattern_compare_3d

## start timer to asses how long process takes
s = perf_counter()

con2d = sqlite3.connect("database_2d.db")
cur2d = con2d.cursor()

## create test data for algorithm development
np.random.seed(42)
rand_array = np.random.random((30, 720, 1280))
z_len = rand_array.shape[0]
y_len = rand_array.shape[1]
x_len = rand_array.shape[2]

dist_sim_threshold = 0.85
centroid_sim_threshold_x = 0.85
centroid_sim_threshold_y = 0.85

# ingest data frame by frame
for frame in range(rand_array.shape[0]):
    #################################################################################
    ####################### FIRST TASK: FIND PATTERNS IN FRAME ######################
    #################################################################################
    frame_patoms = patoms2d(x_len, y_len, rand_array[frame,:,:], frame)
    # patom = [[norm_x, norm_y], [pattern_centroid_x, pattern_centroid_y], patom_ind, frame_ind, patom_time]
    # patom[i][[0][0]: list of x_pos, patom[i][[0][1]: list of y_pos, patom[i][[1][0]: x_cent, patom[i][[1][1]: y_cent, patom[i][[2]: patom_ind, patom[i][[3]: frame_ind, patom[i][[4]: patom_time,
    num_patoms = len(frame_patoms)
    
    #############################################################################################################################################################
    ########## THIRD TASK: STORE NEW PATTERNS IN EMPTY TABLES AND ADD OLD (PATTERNS SIMILAR TO PREVIOUSLY RECEIVED DATA) IN THEIR RESPECTIVE TABLES #############
    #############################################################################################################################################################
    ## get pattern tables and find empty and non-empty
    nonref = cur2d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%ref%'")
    tables_nonref = nonref.fetchall() # returns tuple: (table_name, )
    empty_nonref_tables = []
    nonempty_nonref_tables = []
    for (table,) in tables_nonref:
        cur2d.execute(f"SELECT * FROM {table} LIMIT 1") 
        rows = cur2d.fetchone()
        if rows is None:
            empty_nonref_tables.append(table)
        else:
            nonempty_nonref_tables.append(table)

    ## get reference tables and find empty and non-empty
    ref = cur2d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%ref%';")
    tables_ref = ref.fetchall()
    empty_ref_tables = []
    nonempty_ref_tables = []
    for (table,) in tables_ref:
        cur2d.execute(f"SELECT * FROM {table} LIMIT 1")
        rows = cur2d.fetchone()
        if rows is None:
            empty_ref_tables.append(table)
        else:
            nonempty_ref_tables.append(table)

    # if there are non-empty reference tables then loop through tables and compare reference pattern against newly acquired patterns
    # if simialr patterns exist add to relevant existing table, if no similar patterns exist write new patterns to empty table
    # create list to hold patoms that were added to existing tables
    added_list = []
    if nonempty_ref_tables:
        # compare new patterns against existing reference patterns, if patterns do not match ref pattern store in new table
        patom_ref_indexes = list(product(range(num_patoms), range(len(nonempty_ref_tables))))
        with Pool(processes=cpu_count()) as pool:
            items = [(frame_patoms[i[0]][0], frame_patoms[i[0]][1], frame_patoms[i[0]][2], frame_patoms[i[0]][3],\
                      nonempty_ref_tables[i[1]][0], nonempty_ref_tables[i[1]][1], nonempty_ref_tables[i[1]][2], nonempty_ref_tables[i[1]][3], i) for i in patom_ref_indexes]
            ## function outputs ind value of the patom_indexes list, the centroid and distance similarity measures
            res = pool.starmap(pattern_compare_2d, items)
            print(res[0])
            # res output: [1.0, (0.05407046509274386, 0.03361332388786884)]
            #print("Time to compare 2D patterns with multiprocessing (secs):", (perf_counter()-atime))
            ## loop through the output of the comparison function
            for ix, i in enumerate(res):
                ## pass if its the same patom in the frame being compared against itself
                if patom_ref_indexes[ix][0] == patom_ref_indexes[ix][1]:
                    pass
                ## check if compared patterns fall within similarity threshold values, if they do store in table
                elif (i[1][0] >= centroid_sim_threshold_x) and (i[1][1] >= centroid_sim_threshold_y) and (i[0] >= dist_sim_threshold):
                        # add patom to relevant pattern table get name of ref table to get data table name
                        table_num = nonempty_ref_tables[patom_ref_indexes[ix][1]][-3:] 
                        new_patom = frame_patoms[patom_ref_indexes[ix][0]]
                        patom_table = f"pat_2d_{table_num}"
                        cur2d.executemany(f"INSERT INTO {patom_table}(distance_positions, centroid, patom_ind, frame_ind, patom_time) VALUES (?,?,?,?,?)", new_patom)
                        #patom = [[norm_x, norm_y], [pattern_centroid_x, pattern_centroid_y], patom_ind, frame_ind, patom_time]
                        added_list.append(patom_ref_indexes[ix][0])
                else:
                    pass
            
    # if there are no non-empty reference tables or if none of the patterns are similar to ref patoms then write patterns to new tables
    else:
        # remove patoms from newly acquired data that have already been added to exiiting pattern tables
        if added_list:
            remaining_patoms = [i for j, i in enumerate(frame_patoms) if j not in added_list]
            # convert patom to correct format to insert into table
            tables_to_write_to_db = []
            for i in remaining_patoms:
                pat_len = len(i[0])
                cent_x = [i[2]] * pat_len
                cent_y = [i[3]] * pat_len
                patom_ind = [i[4]] * pat_len
                frame_ind = [i[5]] * pat_len
                patom_time = [i[6]] * pat_len
                patom_to_table = list(zip(i[0], i[1], cent_x, cent_y, patom_ind, frame_ind, patom_time))
                tables_to_write_to_db.append(patom_to_table)
            # compare exisiting patterns and add similar patterns to empty table? later
            for ix, i in enumerate(tables_to_write_to_db):
                pat_len = len(i[0])
                cur2d.executemany(f"INSERT INTO {empty_nonref_tables[i]}(x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time) VALUES (?,?,?,?,?,?,?)", i)
        else:
            tables_to_write_to_db = []
            for i in frame_patoms:
                pat_len = len(i[0])
                cent_x = [i[2]] * pat_len
                cent_y = [i[3]] * pat_len
                patom_ind = [i[4]] * pat_len
                frame_ind = [i[5]] * pat_len
                patom_time = [i[6]] * pat_len
                patom_to_table = list(zip(i[0], i[1], cent_x, cent_y, patom_ind, frame_ind, patom_time))
                tables_to_write_to_db.append(patom_to_table)
            for ix, i in enumerate(tables_to_write_to_db):
                cur2d.executemany(f"INSERT INTO {empty_nonref_tables[ix]}(x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time) VALUES (?,?,?,?,?,?,?)", i)



## save 2D patterns to database and compare new patterns to existing patterns
con2d = sqlite3.connect("database_2d.db")
cur2d = con2d.cursor()


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
