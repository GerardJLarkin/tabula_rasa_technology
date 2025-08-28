## prediction (maybe)
import numpy as np
import os
import glob
import sys
from time import perf_counter
from typing import Iterable, Callable, Generator, Any, Tuple
from typing import Callable, List, Tuple, Set, Dict
from itertools import product, islice
import pickle
import math
from collections import defaultdict
from operator import itemgetter

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.patoms import patoms
from tabula_rasa_technology.simple_approach.unnormalise import unnormalise_xy

root = os.path.dirname(os.path.abspath(__file__))

# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
# sequence = sequence[:100, ...]
# pick 1 sequence from within the training data to assess prediction algorithm
sequence = sequence[12:13, ...]
#print('loaded original data')
# load reference patoms
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname)) for fname in os.listdir(reference)]

## load visual reference linking patoms
with open(root+'/vrlp0.pkl', 'rb') as fp:
    vrlp0 = pickle.load(fp)
with open(root+'/vrlp1.pkl', 'rb') as fp:
    vrlp1 = pickle.load(fp)
with open(root+'/vrlp2.pkl', 'rb') as fp:
    vrlp2 = pickle.load(fp)
with open(root+'/vrlp3.pkl', 'rb') as fp:
    vrlp3 = pickle.load(fp)
with open(root+'/vrlp4.pkl', 'rb') as fp:
    vrlp4 = pickle.load(fp)

# dictionaries in a list to access later
vrlps = [vrlp0, vrlp1, vrlp2, vrlp3, vrlp4]

## load visual refernece linking vectors
with open(root+'/vrlv0.pkl', 'rb') as fv:
    vrlv0 = pickle.load(fv)
with open(root+'/vrlv1.pkl', 'rb') as fv:
    vrlv1 = pickle.load(fv)
with open(root+'/vrlv2.pkl', 'rb') as fv:
    vrlv2 = pickle.load(fv)
with open(root+'/vrlv3.pkl', 'rb') as fv:
    vrlv3 = pickle.load(fv)
with open(root+'/vrlv4.pkl', 'rb') as fv:
    vrlv4 = pickle.load(fv)

vrlvs = [vrlv0, vrlv1, vrlv2, vrlv3, vrlv4]


## visual reference linking patoms
def find_best_matches(arrays, references, compare_func):
    
    matches = set()
    for arr in arrays:
        best_score = float('inf')
        best_ref_id = None
        for ref in references:
            id1, id2, score = compare_func(arr, ref)
            if score < best_score:
                best_score = score
                best_ref_id = id2
                x_cent = arr[0,1] 
                y_cent = arr[0,2]
                seg = arr[0,3]
                min_x = arr[1,0]
                max_x = arr[1,1]
                min_y = arr[1,2]
                max_y = arr[1,3]
                # 0, 1, 2, 3, 4, 5, 6, 7
                best_ref = (best_ref_id, x_cent, y_cent, min_x, max_x, min_y, max_y, seg)
        matches.add(best_ref)
    
    return matches

## set exit frame to allow prediction of next frame
num_frames = 2

## start prediction
st1 = perf_counter()

vrlp_dict = defaultdict(float)
vrlv_dict = defaultdict(float)

# for j in range(20):
#     dict_idx = 5 % (j+1)
#     if j == num_frames:
frame = sequence[0][3]
seq_out_patoms = patoms(frame)
best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare)

current_ref_ids = []
for i in best_matches:
    current_ref_ids.append(i)

vrlp = vrlps[3]
in_scope_keys_vals_sublists = [[] for _ in range(len(current_ref_ids))]
for idx, i in enumerate(current_ref_ids):
    print(i[0])
    for k, v in vrlp.items():
        dict_keys = np.frombuffer(k, dtype=np.float32)
        if i[0] == dict_keys[0]:
            in_scope_keys_vals_sublists[idx].append((dict_keys, v))


for idx, i in enumerate(in_scope_keys_vals_sublists):
    if len(i) == 0:
        continue
    max_key_val_per_ref_id = sorted(i, key=lambda x: x[1], reverse=True)[0][0][1]
    print(max_key_val_per_ref_id)    
print('num ref ids', len(current_ref_ids))
# in_scope_keys_vals_sublists = [[] for _ in range(len(current_ref_ids))]
# seen = {}
# result = []
# for tup in in_scope_keys_vals:
#     key = tup[0][0]
#     if key not in seen:
#         seen[key] = []
#         result.append(seen[key])  # append sublist once
#     seen[key].append(tup)

# sorted_results = []
# for i in result:
#     max_key = sorted(i, key=lambda t: t[1], reverse=True)[0]
#     print(max_key)

# vrlv = vrlvs[dict_idx]
# for k, v in vrlv.items():
#     dict_keys = np.frombuffer(k, dtype=np.float32)
#     keys = str(f'{dict_keys[0]:.8f}')+str(f'{dict_keys[1]:.8f}')
#     vrlv_dict[keys] = v


# next_frame_keys = [(k[11:], v) for k, v in vrlp_dict.items() if k[:11] in current_ref_ids]   
# print(next_frame_keys)

en1 = perf_counter()
print('Time taken to predict 5 frames (mins):', round((en1-st1)/60,4))
