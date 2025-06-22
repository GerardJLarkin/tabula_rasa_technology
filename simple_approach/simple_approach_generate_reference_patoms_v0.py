# generate reference patoms
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
import os
import glob

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from simple_approach_patoms_v0 import patoms
from simple_approach_compare_v0 import compare

# read in to memory all files from disk (only numpy arrays saved in directory)
directory = '/home/gerard/Desktop/capstone_project/simple_approach/historic_data'
file_paths = glob.glob(os.path.join(directory, '*.npy'))
patoms = [np.load(f, allow_pickle=True) for f in file_paths]
print(patoms[0].shape)
