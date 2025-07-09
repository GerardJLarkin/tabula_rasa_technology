# intelligence building
import numpy as np
import os
import sys
from time import perf_counter
from typing import Callable, List, Tuple, Set
from itertools import product
import pickle
from collections import deque

start = perf_counter()


## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')

from tabula_rasa_technology.simple_approach.compare_v1 import compare
from tabula_rasa_technology.simple_approach.patoms_v1 import patoms

root = os.path.dirname(os.path.abspath(__file__))
# load reference patoms
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]

## visual reference linking patoms
ref_ids = [i[0,0] for i in ref_patoms]
vrlp_keys = [i[0]+i[1] for i in product(ref_ids, repeat=2)]
vrlp0 = dict.fromkeys(vrlp_keys, 0) # 61.5MB to start
print('vrlp',sys.getsizeof(vrlp0))
vrlp1 = dict.fromkeys(vrlp_keys, 0)
vrlp2 = dict.fromkeys(vrlp_keys, 0)
vrlp3 = dict.fromkeys(vrlp_keys, 0)
vrlp4 = dict.fromkeys(vrlp_keys, 0)

## visual reference linking vector
segments = [f"{i[0]:0>2}"+f"{i[1]:0>2}" for i in product((range(16)), repeat=2)] # 00 to 15
magnitudes = ['00','01','02','03','04','05','06','07','08','09'] # 00 to 09
vectors = [i[0]+i[1] for i in product(segments, magnitudes)]
vrlv_keys = [i[0]+i[1] for i in product(ref_ids, vectors)]
vrlv0 = dict.fromkeys(vrlv_keys, 0.0) # 1GB to start
print('vrlv',sys.getsizeof(vrlv0))
vrlv1 = dict.fromkeys(vrlv_keys, 0.0)
vrlv2 = dict.fromkeys(vrlv_keys, 0.0)
vrlv3 = dict.fromkeys(vrlv_keys, 0.0)
vrlv4 = dict.fromkeys(vrlv_keys, 0.0)

# del objects to free up memory
del ref_ids; del vrlp_keys; del segments; del magnitudes; del vectors; del vrlv_keys

# use the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# due to memory/processing limitations only use the first 100 of the 10000 total examples.
sequence = sequence[:50, ...]

## visual reference linking patoms
def find_best_matches(
    arrays: List[np.ndarray],
    references: List[np.ndarray],
    compare_func: Callable[[np.ndarray, np.ndarray], Tuple[str, str, float]]) -> Set[Tuple[str, float, float, int]]:
    
    matches: Set[Tuple[str, float, float, int]] = set()
    for arr in arrays:
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
        matches.add((best_ref, segment, cent_x, cent_y))
    return matches

working_memory = deque(maxlen=5)

st1 = perf_counter()
for ix in range(0,50,1):
    s = perf_counter()
    seq = sequence[ix]
    
    prev = None
    for j in range(0,20,1):
        
        frame = seq[j]
        seq_out_patoms = patoms(frame)
        best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare)
        if prev is not None:
            matches = [i[0][0]+i[1][0] for i in product(prev, best_matches)]
            for i in matches:
                vrlp0[i] += 0.0000001
            direction = [f"{i[0][1]:0>2}"+f"{i[1][1]:0>2}" for i in product(prev, best_matches)]
            magnitude = [f"{int(round((np.sqrt((i[0][2] - i[1][2])**2 + (i[0][3]-i[1][3])**2)) / 89.1,1)*10):0>2}" 
                                    for i in product(prev, best_matches)]
            vectors = [a[-6:]+b+c for a, b, c in zip(matches, direction, magnitude)]
            for i in vectors:
                vrlv0[i] += 0.0000001
        prev = best_matches
    e = perf_counter()
    print('seq_num:', ix, 'time taken (mins):', round((e-s)/60,4) )

en1 = perf_counter()
print('Time taken for 100 seqs (mins):', round((en1-st1)/60,4))

# write intelligence to disk
print('vrlp0',sys.getsizeof(vrlp0))
with open('vrlp0.pkl', 'wb') as f:
    pickle.dump(vrlp0, f)
with open('vrlp1.pkl', 'wb') as f:
    pickle.dump(vrlp1, f)
with open('vrlp2.pkl', 'wb') as f:
    pickle.dump(vrlp2, f)
with open('vrlp3.pkl', 'wb') as f:
    pickle.dump(vrlp3, f)
with open('vrlp4.pkl', 'wb') as f:
    pickle.dump(vrlp4, f)

print('vrlv0', sys.getsizeof(vrlv0))
with open('vrlv0.pkl', 'wb') as f:
    pickle.dump(vrlv0, f)
with open('vrlv1.pkl', 'wb') as f:
    pickle.dump(vrlv1, f)
with open('vrlv2.pkl', 'wb') as f:
    pickle.dump(vrlv2, f)
with open('vrlv3.pkl', 'wb') as f:
    pickle.dump(vrlv3, f)
with open('vrlv4.pkl', 'wb') as f:
    pickle.dump(vrlv4, f)