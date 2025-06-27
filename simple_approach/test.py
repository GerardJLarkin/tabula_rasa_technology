# import os
# import numpy as np
# from itertools import combinations
# from multiprocessing import Pool

# intelligence building
import numpy as np
import os
import glob
import sys
from time import perf_counter
from typing import Iterable, Callable, Generator, Any, Tuple
from typing import Callable, List, Tuple, Set
from itertools import product
import pickle
import math
import random
import string

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')

from tabula_rasa_technology.simple_approach.compare_v0 import ref_compare
from tabula_rasa_technology.simple_approach.patoms_v0 import patoms

root = os.path.dirname(os.path.abspath(__file__))
# load reference patoms
# how will this impact on memory? can I hold them all and still save room for processing?
reference = os.path.join(root, 'reference_patoms')
# ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]
# print('reference loaded')

test_dict_ref_patom = dict()
for fname in os.listdir(reference):
    test_dict_ref_patom[(fname.split('.')[0]).split('_')[1]] = np.load(os.path.join(reference, fname), allow_pickle=True)

print(test_dict_ref_patom)

# ref_ids = [i[0,0] for i in ref_patoms]
# segments = [f"{i[0]:0>2}"+f"{i[1]:0>2}" for i in product((range(16)), repeat=2)]
# vec_keys = [i+j for i, j in product(ref_ids, segments)]
# print(vec_keys[:10])

# vrld = dict.fromkeys(vec_keys, 0.0) # 15.4MB to start

# print(sys.getsizeof(vrld))

# # Using the same input dataset as per the compartor CNN-LSTM model
# data = np.load('mnist_test_seq.npy')
# # Swap the axes representing the number of frames and number of data samples.
# dataset = np.swapaxes(data, 0, 1)
# # We'll pick out 1000 of the 10000 total examples and use those.
# dataset = dataset[:100, ...]
# print('loaded')
# #generate patoms from sequences and save to disk
# seq_ind = 0
# for i in range(0,1,1):
#     print('sequnce num:',i)
#     sequence = dataset[i]
#     for j in range(0,20,1):
#         print(sequence[j].shape)


# frame_size = np.zeros((64,64))

# test_data = [] # centroind values will change
# for i in range(6):
#     patom_id = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(6))
#     x_cent = np.random.randint(low=0, high=63, dtype='int')
#     y_cent = np.random.randint(low=0, high=63, dtype='int')
#     id_pat_centroid = (patom_id, x_cent, y_cent)
#     test_data.append(id_pat_centroid)

# print(test_data)
# center_x = (frame_size.shape[0]-1)/2; center_y = (frame_size.shape[1]-1)/2 # scalar will never change
# num_segments = 16 # scalar will never change
# segment_width = 360 / num_segments
# print(center_x, center_y)
# initial_segments = []
# for i in test_data:
#     x = i[1]; y = i[2]
#     angle_deg = (np.degrees(np.arctan2(center_y - y, x - center_x)) + 360) % 360
#     angle_clockwise_from_north = (90 - angle_deg) % 360
#     segment = f"{(angle_clockwise_from_north // segment_width).astype(int):0>2}"
#     initial_segments.append(segment)

# print(initial_segments)

# def find_vector(data):
#     for a, b, c in zip((64,64), data, data[1:], data[2:]):
#         segment = 360 / 16
#         center_x = math.floor((64-1)/2); center_y = math.floor((64-1)/2)
#         strt_dx1 = a[1] - center_x
#         strt_dy1 = center_y - a[2]
#         start_seg_1 = f"{int((np.degrees(np.arctan2(strt_dx1, strt_dy1)) % 360) // segment):0>2}"
#         dx1 = b[1] - a[1]
#         dy1 = b[2] - a[2]
#         seg_1 = f"{int((np.degrees(np.arctan2(dy1, dx1)) % 360) // segment):0>2}"
#         start_seg_2 = f"{int((np.degrees(np.arctan2(b[1], b[2])) % 360) // segment):0>2}"
#         dx2 = c[1] - b[1]
#         dy2 = c[2] - b[2]
#         seg_2 = f"{int((np.degrees(np.arctan2(dy2, dx2)) % 360) // segment):0>2}"
#         print(f"F2F vector direction: {a[0]}→{b[0]} = {start_seg_1+seg_1}")
#         print(f"F2F vector direction: {b[0]}→{c[0]} = {start_seg_2+seg_2}")


# find_vector(test_data)