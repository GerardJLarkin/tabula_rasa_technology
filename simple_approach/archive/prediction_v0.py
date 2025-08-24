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
print(ref_patoms[:2])
print('loaded reference patoms')

## load visual reference linking patoms
with open(root+'/vrlp.pkl', 'rb') as fp:
    vrlp = pickle.load(fp)

## load visual refernece linking vectors
with open(root+'/vrlv.pkl', 'rb') as fv:
    vrlv = pickle.load(fv)

## visual reference linking patoms
def find_best_matches(arrays, references, compare_func):
    
    matches = list()
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
    prev_cent = None
    for j in range(0,10,1):
        frame = seq[j]
        print('frame:',j)
        seq_out_patoms = patoms(frame)
        print('num patoms:',len(seq_out_patoms))
        centroids = [(i[0,6],i[0,7]) for i in seq_out_patoms]
        print('patom centroids:',centroids)
        best_matches = find_best_matches(seq_out_patoms, ref_patoms, ref_compare)
        if prev is not None:
            for pat1 in prev:
                loop_list = []
                print('--loop--')
                for pat2 in best_matches:
                    centroid_dist = np.sqrt((pat2[2] - pat1[2])**2 + (pat2[3]-pat1[3])**2)
                    direction = f"{pat1[1]:0>2}"+f"{pat2[1]:0>2}"
                    magnitude = f"{int(round((np.sqrt((pat2[2] - pat1[2])**2 + (pat2[3]-pat1[3])**2)) / 89.1,1)*10):0>2}" 
                    loop_list.append((pat1[0]+pat2[0], centroid_dist, direction, magnitude))
                frame_compare_ref_patoms = min(loop_list, key=lambda i:i[1])
                print(frame_compare_ref_patoms)
            # frame_compare = list(product(prev, best_matches))
            # for patom_compare in frame_compare:
            #     print(patom_compare)
                # ids = [i[0][0]+i[1][0] for i in patom_compare]         
                # frame_0_ids = [i[0][0] for i in patom_compare] 
                # centroid_dist = [np.sqrt((i[0][2] - i[1][2])**2 + (i[0][3]-i[1][3])**2) for i in patom_compare]
                # print('ids cent dist', sorted(list(zip(frame_0_ids, centroid_dist, ids))))

                # direction = [f"{i[0][1]:0>2}"+f"{i[1][1]:0>2}" for i in patom_compare]
                # # review the output of this
                # magnitude = [f"{int(round((np.sqrt((i[0][2] - i[1][2])**2 + (i[0][3]-i[1][3])**2)) / 89.1,1)*10):0>2}" 
                #                         for i in patom_compare]
                # vectors = [a[-6:]+b+c for a, b, c in zip(ids, direction, magnitude)]
                # print('cross join best matches vectors',vectors)
            

        prev = best_matches
        prev_cent = centroids

en1 = perf_counter()
print('Time taken for 100 seqs (mins):', round((en1-st1)/60,4))






# cnt = 0
# while cnt < 100:
#     for key, value in vrlv.items():
#         if value > 0:
#             print(key, value)
#             cnt += 1
