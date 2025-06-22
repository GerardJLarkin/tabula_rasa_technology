# simple approach
# import all libraries
import numpy as np
import cv2 as cv
from operator import itemgetter
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
from multiprocessing import Pool, cpu_count
import sqlite3
import random
import sys
import random
import string

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
sys.path.append('/home/gerard/Desktop/capstone_project')

# import local functions
from simple_approach_patoms_v0 import patoms
from simple_approach_compare_v0 import compare

#############################################################################################################
#############################################################################################################
# generate pseduo real world data
image = np.zeros((30, 30, 3))
circle = cv.circle(image, (15,15), 10, (255,255,255), -1)
# in an ideal setting I would merge each of the 3 channels to create a pseudo 9 digit value representative
# of the treu colour where the first 3 digits for the red channel, the second 3 digits for the green channel
# and the third 3 digits for the blue channel
flat_circle = circle[:,:,0]

# set array pad values (based on image size) which result in a final array size
top = 450
bottom = 10
left = 10
right = 450
padded_circle = np.pad(flat_circle, ((top, bottom), (left, right)), 'constant', constant_values=(0)) 
# pad array to get a standard HD aspect ratio, I create mutiple arrays slightly moving the circle in each array
# giving a sense of the circle moving over time
rolled_arrays = []
# this loop creates 150 snapshots (5 seconds) worth of pseudo data, with the circle having an apparent movement 
# across the array diagonally
for i in range(0, 450, 3):
    rolled_array_x = np.roll(padded_circle, i, axis=0)
    rolled_array_xy = np.roll(rolled_array_x, i, axis=1)
    rolled_arrays.append(rolled_array_xy)

#############################################################################################################
#############################################################################################################
# set up empty list to hold patoms based on sequence id
seq_0_patoms = []
seq_1_patoms = []
# get patoms from data
seq_ind = 0
for ix, i in enumerate(rolled_arrays):
    out_patoms = patoms(i, seq_ind, ix)
    if seq_ind == 0:
        seq_0_patoms.append(patoms)
        seq_ind = 1
    elif seq_ind == 1:
        seq_1_patoms.append(patoms)
        seq_ind = 0
