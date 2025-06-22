## add ignore warnings for now, will remove and debug once full algorithm is complete
# import warnings
# warnings.filterwarnings("ignore")

## import packages/libraries
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import numpy as np
from multiprocessing import Pool, cpu_count
import sys
import cv2 as cv
import random
import sys
import random
import string
import math
import gc
from itertools import product

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from simple_approach_patoms_v0 import patoms
from simple_approach_compare_v0 import compare

# generate pseduo real world data
# create a numpy array of zeros on which to place the shapes created below
image = np.zeros((30, 30, 3))
# in an ideal setting I would merge each of the 3 channels to create a pseudo 9 digit value representative
# of the treu colour where the first 3 digits for the red channel, the second 3 digits for the green channel
# and the third 3 digits for the blue channel
shapes = []
circle = cv.circle(image, (15,15), 10, (255,255,255), -1)
flat_circle = circle[:,:,0]
rectangle1 = cv.rectangle(image, (5, 5), (25, 25), (0, 0, 255), -1)
flat_rectangle1 = rectangle1[:,:,2]
rectangle2 = cv.rectangle(image, (5, 10), (27, 29), (0, 0, 200), -1)
flat_rectangle2 = rectangle2[:,:,2]
ellipse1 = cv.ellipse(image, (15, 15), (27, 20), 20, 0, 360, (0, 255, 0), -1)
flat_ellipse1 = ellipse1[:,:,1]
ellipse2 = cv.ellipse(image, (15, 15), (15, 25), 60, 0, 360, (255, 0, 0), -1)
flat_ellipse2 = ellipse2[:,:,1]
points1 = np.array([[9, 4], [26, 9], [14, 23]])
polygon1 = cv.fillPoly(image, pts=[points1], color=(255, 0, 0))
flat_polygon1= polygon1[:,:,1]
points2 = np.array([[19, 4], [5, 9], [26, 11], [3, 17], [24, 20], [17, 24]])
polygon2 = cv.fillPoly(image, pts=[points2], color=(255, 0, 0))
flat_polygon2= polygon2[:,:,1]

# set array pad values (based on image size) which result in a final array size
top = 450
bottom = 10
left = 10
right = 450
padded_circle = np.pad(flat_circle, ((top, bottom), (left, right)), 'constant', constant_values=(0)); shapes.append(padded_circle)
padded_rec1 = np.pad(flat_rectangle1, ((top, bottom), (left, right)), 'constant', constant_values=(0)); shapes.append(padded_rec1)
padded_rec2 = np.pad(flat_rectangle2, ((top, bottom), (left, right)), 'constant', constant_values=(0)); shapes.append(padded_rec2)
padded_elp1 = np.pad(flat_ellipse1, ((top, bottom), (left, right)), 'constant', constant_values=(0)); shapes.append(padded_elp1)
padded_elp2 = np.pad(flat_ellipse2, ((top, bottom), (left, right)), 'constant', constant_values=(0)); shapes.append(padded_elp2)
padded_poly1 = np.pad(flat_polygon1, ((top, bottom), (left, right)), 'constant', constant_values=(0)); shapes.append(padded_poly1)
padded_poly2 = np.pad(flat_polygon2, ((top, bottom), (left, right)), 'constant', constant_values=(0)); shapes.append(padded_poly2)
# pad array to get a standard HD aspect ratio, I create mutiple arrays slightly moving the circle in each array
# giving a sense of the circle moving over time
array_sequence_set = []
# this loop creates 150 snapshots (5 seconds) worth of pseudo data, with the circle having an apparent movement 
# across the array
for shape in shapes:
    rolled_arrays0 = []
    for i in range(0, 450, 3):
        rolled_arrays = []
        rolled_array_x = np.roll(shape, i, axis=0)
        rolled_array_xy = np.roll(rolled_array_x, i, axis=1)
        rolled_arrays0.append(rolled_array_xy)

    array_sequence_set.append(rolled_arrays0)
    
    rolled_arrays1 = []
    for i in range(0, 450, 3):
        rolled_arrays = []
        rolled_array_x = np.roll(shape, i, axis=0)
        rolled_arrays1.append(rolled_array_x)

    array_sequence_set.append(rolled_arrays1)

    rolled_arrays2 = []
    for i in range(0, 450, 3):
        rolled_arrays = []
        rolled_array_y = np.roll(shape, i, axis=1)
        rolled_arrays2.append(rolled_array_y)

    array_sequence_set.append(rolled_arrays2)
    
    rolled_arrays3 = []
    for i in range(0, 450, 3):
        rolled_arrays = []
        rolled_array_x = np.roll(shape, math.floor(i/4), axis=0)
        rolled_array_xy = np.roll(rolled_array_x, i, axis=1)
        rolled_arrays3.append(rolled_array_xy)

    array_sequence_set.append(rolled_arrays3)
    
    rolled_arrays4 = []
    for i in range(0, 450, 3):
        rolled_arrays = []
        rolled_array_x = np.roll(shape, math.floor(i/3), axis=0)
        rolled_array_xy = np.roll(rolled_array_x, i, axis=1)
        rolled_arrays4.append(rolled_array_xy)

    array_sequence_set.append(rolled_arrays4)
    
    rolled_arrays5 = []
    for i in range(0, 450, 3):
        
        rolled_array_x = np.roll(shape, math.floor(i/2), axis=0)
        rolled_array_xy = np.roll(rolled_array_x, i, axis=1)
        rolled_arrays5.append(rolled_array_xy)

    array_sequence_set.append(rolled_arrays5)

# generate patoms from sequences and save to disk
seq_ind = 0
for sx, shape_sequence in enumerate(array_sequence_set[:1]):
    for ix, frame in enumerate(shape_sequence):
        if seq_ind == 0:
            out_patoms0 = patoms(frame, seq_ind, ix)
            for i in out_patoms0:
                patom_id = i[0,0]
                # save patoms to disk
                np.save(f'historic_data/patom_{patom_id}', i)
            seq_ind = 1
            del out_patoms0
        elif seq_ind == 1:
            out_patoms1 = patoms(frame, seq_ind, ix)
            for i in out_patoms1:
                patom_id = i[0,0]
                # save patoms to disk
                np.save(f'historic_data/patom_{patom_id}', i)
            seq_ind = 0
            del out_patoms1


end = perf_counter()
print("Time taken (mins):", (end - start)/60)