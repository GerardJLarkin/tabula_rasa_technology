###### realtime data test run
## add ignore warnings for now, will remove and debug once full algorithm is complete
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool, cpu_count
import sys
import sqlite3
import cv2 as cv

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

## call locally created functions
from snapshot_2d_pattern_v7 import patoms2d
from pattern_2d_compare_v6 import pattern_compare_2d
from updating_ref_table_v0 import update_ref_table

## access the camera to get video stream
cap = cv.VideoCapture(0)

s = perf_counter()
val = 0
while val <= 4:
    # frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8) # 307200 bytes
    # print(frame.nbytes)
    ret, frame = cap.read()
    frame = (frame[..., 0] << 16) | (frame[..., 1] << 8) | frame[..., 2]
    frame = (frame - 127.5) / 127.5
    x_len = frame.shape[0]
    y_len = frame.shape[1]
    ####################### FIRST TASK: FIND PATTERNS IN FRAME ######################
    frame_patoms = patoms2d(x_len, y_len, frame, val)
    # output: [norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, quad, quad_cnt, patom_ind, frame_ind_arr, patom_time]
    #frame_patoms_arr = np.vstack((frame_patoms))
    num_patoms = len(frame_patoms)
    #print(num_patoms)
    ############## SECOND TASK: COMPARE NEW PATOMS AGAINST REF PATOMS ###############
    # atime = perf_counter()
    # with Pool(processes=8) as pool:
    #     #indices = list(product(range(num_patoms), range(len(working_ref_patoms))))
    #     items = [(frame_patoms[i], ref_patoms_array) for i in range(num_patoms)]
    #     comp_results = pool.starmap(pattern_compare_2d, items) #e.g. ['pcol','px','py','pxc','pyc','pq','pqlen','pfind','xc_d','yc_d','x_d','y_d','rfind','similar']
    #     print("Time to compare 2D patterns with multiprocessing (mins):", (perf_counter()-atime)/60)
        
    #     ## loop through the output of the comparison function
    #     witime = perf_counter()
    #     # for ix, i in enumerate(comp_results):
    #     #     existing_tables = [names for (names,) in cur2d.execute("select name from sqlite_master where type='table';").fetchall()]
    #     #     next_table_num = int(existing_tables.pop(-1)[-6:]) + 1 
    #     #     next_table_num = str(next_table_num).zfill(6)
    #     #     ref_patom_ind = i[:,13]
    #     #     ref_patom_ind = np.unique(ref_patom_ind).tolist()
    #     #     print(ref_patom_ind)
    #         # extract reference tables that are similar to the patom in the current iteration
    #         # if ref_patom_ind:
    #         #     for i in ref_patom_ind:
    #         #         print(i)
    #         #         table_num = ref_names[int(i)][-6:]
    #         #         print(table_num)
    #         #     cur2d.executemany(f"INSERT INTO pat_2d_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, frame_ind, \
    #         #                         xc_d, yc_d, x_d, y_d) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", i)
    #         # # if no matching ref patoms, write new patom to new table
    #         # else:
    #         #     cur2d.execute(f"CREATE TABLE pat_2d_{next_table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, frame_ind, \
    #         #                   xc_d, yc_d, x_d, y_d);")
    #         #     cur2d.executemany(f"INSERT INTO pat_2d_{next_table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, quad, quad_cnt, frame_ind, \
    #         #                       xc_d, yc_d, x_d, y_d) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", i)    
            
    #         # con2d.commit()
    #     ## major bottleneck is the insert/create steps
    #     print("Time to write/insert patoms to existing/new tables in database (secs):", (perf_counter()-witime))
    val += 1

print("Time to process 1 second of data against 50 reference patoms(mins):", (perf_counter()-s)/60)