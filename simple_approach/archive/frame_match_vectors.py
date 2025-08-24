import numpy as np
import os
import sys
from time import perf_counter
from typing import Callable, List, Tuple, Set
from itertools import product
import pickle
from collections import defaultdict

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')

from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.patoms import patoms

# find best matches
def find_best_matches(arrays,references,compare_func):
    
    matches = set()
    for arr in arrays:
        best_score = float('inf')
        best_ref_id: float = None
        for ref in references:
            id1, id2, score = compare_func(arr, ref)
            if score < best_score:
                best_score = score
                best_ref_id = id2
                x_cent = arr[0,1] 
                y_cent = arr[0,2]
                seg = arr[0,3]
                best_ref = (best_ref_id, x_cent, y_cent, seg)
        matches.add(best_ref)
    
    return matches

sequence = []
ref_patoms = []
group_dict0 = defaultdict(float)
vrlp0 = defaultdict(float)
vrlv0 = defaultdict(float)

# start function here, input is sequence index and frame, output is list of ref ids (byte array), vrlp keys (byte array), vrlv keys (byte array)
def frame_match_vectors(ind, frame):
    pass

st1 = perf_counter()
for ix in range(0,n,1):
    s = perf_counter()
    seq = sequence[ix]
    
    prev0 = None; prev1 = None; prev2 = None; prev3 = None; prev4 = None
    for j in range(0,20,1):
        if (j < 4) | (j > 20):
            continue
        else:
            frame0 = seq[j-4]
            # start function here, input is sequence index and frame, output is list of ref ids (byte array), vrlp keys (byte array), vrlv keys (byte array)
            results = frame_match_vectors(j, frame0)
            seq_out_patoms0 = patoms(frame0)
            best_matches0 = find_best_matches(seq_out_patoms0, ref_patoms, compare)
            
            ref_ids0 = np.array(sorted([i[0] for i in best_matches0]), dtype=np.float32).tobytes() # consider how to recover the ids?
            group_dict0[ref_ids0] += 0.0000001
            
            if prev0 is not None:
                cross0 = [i for i in product(prev0, best_matches0)]
                
                matches0 = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross0] # length of each byte object is 2
                # mat_test = [np.array([i[0][0], i[1][0]]) for i in cross0]
                for i in matches0:
                    vrlp0[i] += 0.0000001
                
                direction0 = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross0] # length of each byte object is 2
                # dir_test = [np.array([i[0][3], i[1][3]]) for i in cross0]
                magnitude0 = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross0] # length of each byte object is 1
                # mag_test = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1]) for i in cross0]
                vectors0 = [a + b + c for a, b, c in zip(matches0, direction0, magnitude0)]
                # vec_test = [[a, b, c] for a, b, c in zip(mat_test, dir_test, mag_test)]; print('vec_test',vec_test[0])
                
                # lengths = [2, 2, 1]
                # series_list = []
                for i in vectors0:
                    vrlv0[i] += 0.0000001

                #     offset = 0
                #     for count in lengths:
                #         n_bytes = count * 4
                #         chunk = i[offset:offset + n_bytes]
                #         labels = np.frombuffer(chunk, dtype=np.float32)
                #         series_list.append(labels)
                #         offset += n_bytes
                # print('ser_list',series_list[0:3])
                # print('exact match', vec_test[0][0] == series_list[0][0])
            
            prev0 = best_matches0