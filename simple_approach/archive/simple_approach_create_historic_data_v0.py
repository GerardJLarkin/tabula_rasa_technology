## add ignore warnings for now, will remove and debug once full algorithm is complete
# import warnings
# warnings.filterwarnings("ignore")

## import packages/libraries
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import numpy as np
from multiprocessing import Pool, cpu_count
import sys
import sqlite3
import cv2 as cv
import random
from PIL import Image
import sys
import random
import string
import math
import gc

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from simple_approach_patoms_v0 import patoms

## create in memory 2d database
con2d = sqlite3.connect("reference_database.db")
cur2d = con2d.cursor()

# generate pseduo real world data
# create a numpy array of zeros on which to place the shapes created below
image = np.zeros((100, 100, 3))
# in an ideal setting I would merge each of the 3 channels to create a pseudo 9 digit value representative
# of the treu colour where the first 3 digits for the red channel, the second 3 digits for the green channel
# and the third 3 digits for the blue channel
shapes = []
circle = cv.circle(image, (50,50), 30, (255,255,255), -1)
flat_circle = circle[:,:,0]; shapes.append(flat_circle)
rectangle1 = cv.rectangle(image, (25, 25), (75, 75), (0, 0, 255), -1)
flat_rectangle1 = rectangle1[:,:,2]; shapes.append(flat_rectangle1)
rectangle2 = cv.rectangle(image, (20, 30), (80, 50), (0, 0, 200), -1)
flat_rectangle2 = rectangle2[:,:,2]; shapes.append(flat_rectangle2)
ellipse1 = cv.ellipse(image, (50, 50), (80, 60), 20, 0, 360, (0, 255, 0), -1)
flat_ellipse1 = ellipse1[:,:,1]; shapes.append(flat_ellipse1)
ellipse2 = cv.ellipse(image, (50, 50), (90, 20), 60, 0, 360, (255, 0, 0), -1)
flat_ellipse2 = ellipse2[:,:,1]; shapes.append(flat_ellipse2)
points1 = np.array([[30, 60], [50, 20], [70, 60]])
polygon1 = cv.fillPoly(image, pts=[points1], color=(255, 0, 0))
flat_polygon1= polygon1[:,:,1]; shapes.append(flat_polygon1)
points2 = np.array([[30, 30], [50, 30], [70, 50], [20, 60], [40, 80], [60, 80]])
polygon2 = cv.fillPoly(image, pts=[points2], color=(255, 0, 0))
flat_polygon2= polygon2[:,:,1]; shapes.append(flat_polygon2)

# set array pad values (based on image size) which result in a final array size
top = 1810
bottom = 10
left = 10
right = 970
# pad array to get a standard HD aspect ratio, I create mutiple arrays slightly moving the circle in each array
# giving a sense of the circle moving over time
array_sequence_set = []
# this loop creates 150 snapshots (5 seconds) worth of pseudo data, with the circle having an apparent movement 
# across the array
time_spent = 0
while time_spent < 10:
    time_int = int(str(clock_gettime_ns(CLOCK_REALTIME))[-1])
    if time_int == 1:
        padded_arrays = []
        for ix, i in enumerate(range(0, 900, 6)):
            # shift image diagonally1
            pad_array = np.pad(flat_circle, ((top-i, bottom+i), (left+i, right-i)), 'constant', constant_values=(0))
            padded_arrays.append(pad_array)
            #size = sys.getsizeof(padded_arrays)
            #print(size)
            #print(ix, pad_array.shape)
    
        array_sequence_set.append(padded_arrays)
        del padded_arrays
    
    elif time_int == 2:
        padded_arrays = []
        for ix, i in enumerate(range(0, 900, 6)):
            # shift image laterally
            pad_array = np.pad(flat_circle, ((top, bottom), (left+i, right-i)), 'constant', constant_values=(0))
            padded_arrays.append(pad_array)
    
        array_sequence_set.append(padded_arrays)
        del padded_arrays

    elif time_int == 3:
        padded_arrays = []
        for ix, i in enumerate(range(0, 900, 6)):
            # shift image upwards
            pad_array = np.pad(flat_circle, ((top-i, bottom+i), (left, right)), 'constant', constant_values=(0))
            padded_arrays.append(pad_array)
    
        array_sequence_set.append(padded_arrays)
        del padded_arrays
    
    elif time_int == 4:
        padded_arrays = []
        for ix, i in enumerate(range(0, 900, 6)):
            # shift image diagonally2
            pad_array = np.pad(flat_circle, ((top-i, bottom+i), (left+math.floor(i/2), right-math.floor(i/2))), 'constant', constant_values=(0))
            padded_arrays.append(pad_array)
    
        array_sequence_set.append(padded_arrays)
        del padded_arrays
    
    elif time_int == 5:
        padded_arrays = []
        for ix, i in enumerate(range(0, 900, 6)):
            # shift image diagonally3
            pad_array = np.pad(flat_circle, ((top-i, bottom+i), (left+math.floor(i/3), right-math.floor(i/3))), 'constant', constant_values=(0))
            padded_arrays.append(pad_array)
    
        array_sequence_set.append(padded_arrays)
        del padded_arrays

    elif time_int == 6:
        padded_arrays = []
        for ix, i in enumerate(range(0, 900, 6)):
            # shift image diagonally4
            pad_array = np.pad(flat_circle, ((top-math.floor(i/2), bottom+math.floor(i/2)), (left+i, right-i)), 'constant', constant_values=(0))
            padded_arrays.append(pad_array)
    
        array_sequence_set.append(padded_arrays)
        del padded_arrays

    elif time_int == 7:
        padded_arrays = []
        for ix, i in enumerate(range(0, 900, 6)):
            # shift image diagonally5
            pad_array = np.pad(flat_circle, ((top-math.floor(i/3), bottom+math.floor(i/3)), (left+i, right-i)), 'constant', constant_values=(0))
            padded_arrays.append(pad_array)
    
        array_sequence_set.append(padded_arrays)
        del padded_arrays
    
    time_spent += 1

# size = sys.getsizeof(array_sequence_set)
# print(size)
for ix, i in enumerate(array_sequence_set):
    for jx, j in enumerate(i): 
        print(ix, jx, j.shape)

# table_num_seq = 0
# seq_ind = 0
# patoms = []
# for j in padded_arrays_set:
#     for i in j[0:1]:
#         x_len = i.shape[0]; y_len = i.shape[1]
#         if seq_ind == 0:
#             seq_ind = 1
#         elif seq_ind == 1:
#             seq_ind = 0
#         patoms = patoms(x_len, y_len, i, seq_ind)
#         # patoms return a list of numpy arrays: ((patom_id, x_vals, y_vals, norm_x, norm_y, colours, segment, sequence_id))
#         for ind, patom in enumerate(patoms):
#             if ind == 0:
#                 print(ind, patom.shape, patom.min(), patom.max())
#             # print(ind)
#             # diff_cols = np.resize(np.array([0.0,0.0,0.0,0.0,0.0]), (patom.shape[0],5))
#             # nonref_patom = np.hstack((patom, diff_cols))
#             # table_num = str(table_num_seq).zfill(6)
#             # cur2d.execute(f"CREATE TABLE ref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind);")
#             # cur2d.executemany(f"INSERT INTO ref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind) VALUES (?,?,?,?,?,?,?,?)", patom)
#             # cur2d.execute(f"CREATE TABLE nonref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind, col_d, xc_d, yc_d, x_d, y_d);")
#             # cur2d.executemany(f"INSERT INTO nonref_{table_num}(colour, x_pos_dist, y_pos_dist, x_cent, y_cent, segment, segment_cnt, sequence_ind, col_d, xc_d, yc_d, x_d, y_d) \
#             #                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", nonref_patom)
#             table_num_seq += 1 

# con2d.commit() 
# # print(max([table for (table,) in cur2d.execute("select name from sqlite_master where type='table';").fetchall()]))
# # print(np.array(cur2d.execute("select * from ref_000555;").fetchall()))

# # table_names = [table for (table,) in cur2d.execute("select name from sqlite_master where type='table';").fetchall()]
# # table_rows = []
# # for i in table_names:
# #     row_count = cur2d.execute(f"select count(*) from {i};").fetchall()[0][0]
# #     table_rows.append(row_count)
# # print(min(table_rows), max(table_rows))
# con2d.close()