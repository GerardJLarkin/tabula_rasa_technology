# intelligence building
import numpy as np
import os
import glob
import sys
from time import perf_counter
from typing import Iterable, Callable, Generator, Any, Tuple
from typing import Callable, List, Tuple, Set, Dict
from itertools import product, islice
import pickle

start = perf_counter()


## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')

from simple_approach_compare_v2 import ref_compare
from simple_approach_patoms_v1 import patoms

root = os.path.dirname(os.path.abspath(__file__))
# load reference patoms
# how will this impact on memory? can I hold them all and still save room for processing?
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]
print('reference loaded')

# vrllut: ref_patom_id, following ref patom_id, frequency of second ref_patom _id follows first
# key: (ref_patom_id, following_ref_patom_id) value: frequency

## visual reference linking patoms
ref_ids = [i[0,0] for i in ref_patoms]
vrlp_keys = [i[0]+i[1] for i in product(ref_ids, repeat=2)]
vrlp = dict.fromkeys(vrlp_keys, 0) # 61.5MB to start
print(sys.getsizeof(vrlp))

## visual reference linking vector
segments = [f"{i[0]:0>2}"+f"{i[1]:0>2}" for i in product((range(16)), repeat=2)] # 00 to 15
magnitudes = ['00','01','02','03','04','05','06','07','08','09'] # 00 to 09
vectors = [i[0]+i[1] for i in product(segments, magnitudes)]
vrlv_keys = [i[0]+i[1] for i in product(ref_ids, vectors)]
vrlv = dict.fromkeys(vrlv_keys, 0.0) # 1GB to start
print(sys.getsizeof(vrlv))

# del objects to free up memory
del ref_ids; del vrlp_keys; del segments; del magnitudes; del vectors; del vrlv_keys

# use the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# due to memory/processing limitations only use the first 100 of the 10000 total examples.
sequence = sequence[:100, ...]
print('sequence loaded')

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

st1 = perf_counter()
seq_ind = 0
for ix in range(0,100,1):
    s = perf_counter()
    seq = sequence[ix]
    prev = None
    for j in range(0,20,1):
        frame = seq[j]
        seq_out_patoms = patoms(frame, seq_ind)
        best_matches = find_best_matches(seq_out_patoms, ref_patoms, ref_compare)
        if prev is not None:
            matches = [i[0][0]+i[1][0] for i in product(prev, best_matches)]
            for i in matches:
                vrlp[i] += 0.0000001
            direction = [f"{i[0][1]:0>2}"+f"{i[1][1]:0>2}" for i in product(prev, best_matches)]
            magnitude = [f"{ int(round((np.sqrt((i[0][2] - i[1][2])**2 + (i[0][3]-i[1][3])**2)) / 89.1,1)*10):0>2}" 
                                    for i in product(prev, best_matches)]
            vectors = [a[-6:]+b+c for a, b, c in zip(matches, direction, magnitude)]
            for i in vectors:
                vrlv[i] += 0.0000001
        prev = best_matches
    e = perf_counter()
    print('seq_num:', ix, 'time taken (mins):', round((e-s)/60,4) )

en1 = perf_counter()
print('Time taken for Patoms 100 seqs (mins):', round((en1-st1)/60,4))

# write intelligence to disk
print(sys.getsizeof(vrlp))
with open('vrlp.pkl', 'wb') as f:
    pickle.dump(vrlp, f)

print(sys.getsizeof(vrlv))
with open('vrlp.pkl', 'wb') as f:
    pickle.dump(vrlv, f)