## prediction (maybe)
import numpy as np
import os
import glob
import sys
from time import perf_counter
from typing import Iterable, Callable, Generator, Any, Tuple
from typing import Callable, List, Tuple, Set
from itertools import product, islice
import pickle

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.compare_v0 import ref_compare, compare
from tabula_rasa_technology.simple_approach.patoms_v0 import patoms

root = os.path.dirname(os.path.abspath(__file__))

# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
# sequence = sequence[:100, ...]
# pick 1 sequence from within the training data to assess prediction algorithm
sequence = sequence[49:50, ...]
print('loaded original data')
# load reference patoms
reference = os.path.join(root, 'reference_patoms')
dict_ref_patom = dict()
for fname in os.listdir(reference):
    dict_ref_patom[(fname.split('.')[0]).split('_')[1]] = np.load(os.path.join(reference, fname), allow_pickle=True)
    
#ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]
ref_patoms = list(dict_ref_patom.values())
print('loaded reference patoms')

## load visual reference linking patoms
with open(root+'/vrlp.pkl', 'rb') as fp:
    vrlp = pickle.load(fp)

## load visual refernece linking vectors
with open(root+'/vrlv.pkl', 'rb') as fv:
    vrlv = pickle.load(fv)

## visual reference linking patoms
def find_best_matches(
    arrays: List[np.ndarray],
    references: List[np.ndarray],
    compare_func: Callable[[np.ndarray, np.ndarray], Tuple[str, str, float]]) -> List[List]:
    
    matches: List[List] = list()
    for ix, arr in enumerate(arrays):
        #print('patom num:', ix, 'patom id', arr[0,0])
        segment = arr[0,8]
        cent_x = arr[0,6]
        cent_y = arr[0,7]
        best_score = float('inf')
        best_ref: str = None
        for ref in references:
            id1, id2, score = compare_func(arr, ref)
            if score < best_score:
                best_score = score
                best_ref = id2
                #print('patom id',id1,'best matched ref patom id',best_ref, 'score',score)
        matches.append([best_ref, segment, cent_x, cent_y])
    return matches

st1 = perf_counter()
for ix in range(0,1,1):
    s = perf_counter()
    seq = sequence[ix]
    prev = None
    final_frame_compare_output = []
    num_frames = 10
    for j in range(0,num_frames,1):
        frame = seq[j]
        #print('frame:',j)
        seq_out_patoms = patoms(frame)
        best_matches = find_best_matches(seq_out_patoms, ref_patoms, ref_compare)
        if prev is not None:
            for pat1 in prev:
                loop_list = []
                #print('--loop--')
                for pat2 in best_matches:
                    centroid_dist = np.sqrt((pat2[2] - pat1[2])**2 + (pat2[3]-pat1[3])**2)
                    direction = f"{pat1[1]:0>2}"+f"{pat2[1]:0>2}"
                    magnitude = f"{int(round((np.sqrt((pat2[2] - pat1[2])**2 + (pat2[3]-pat1[3])**2)) / 89.1,1)*10):0>2}" 
                    loop_list.append((pat2[0], centroid_dist, direction, magnitude, pat2[2], pat2[3]))
                frame_compare_ref_patoms = min(loop_list, key=lambda i:i[1])
                frame_compare_ref_patoms = frame_compare_ref_patoms[:1] + frame_compare_ref_patoms[2:]
                if num_frames == j+1:
                    final_frame_compare_output.append(frame_compare_ref_patoms)
                    #print(frame_compare_ref_patoms)
        prev = best_matches

    print(final_frame_compare_output)
    # create vectors
    # get next set of patoms
    curr_seq_ref_patoms = [i[0] for i in final_frame_compare_output]
    print(curr_seq_ref_patoms)
    # find all next sequence ref patoms based on curr sequence ref patoms
    next_seq_ref_patoms = []
    for i in curr_seq_ref_patoms:
        next_seq_keys = [(i, k, v) for k, v in vrlp.items() if k[:6] == i]
        next_seq_key = max(next_seq_keys, key=lambda i:i[2])
        next_seq_ref_patoms.append(next_seq_key[1][-6:])
    print(next_seq_ref_patoms)
    # rebuld array to print as image
    # need translation vector
    # need centroid from set of patoms from last input frame

    
en1 = perf_counter()
print('Time taken for 100 seqs (mins):', round((en1-st1)/60,4))






# cnt = 0
# while cnt < 100:
#     for key, value in vrlv.items():
#         if value > 0:
#             print(key, value)
#             cnt += 1
