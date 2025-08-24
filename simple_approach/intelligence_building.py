# intelligence building
import numpy as np
import os
import sys
from time import perf_counter
from itertools import product
import pickle
from collections import defaultdict

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')

from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.patoms import patoms

root = os.path.dirname(os.path.abspath(__file__))
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname)) for fname in os.listdir(reference)]

# initialise dictionaries
vrlps = [defaultdict(float) for _ in range(5)]
vrlvs = [defaultdict(float) for _ in range(5)]
group_dicts = [defaultdict(float) for _ in range(5)]
sequence_dict = defaultdict(float)
sequence_dict_len = defaultdict(list)

# use the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# due to memory/processing limitations only use the first 100 of the 10000 total examples.
n = 25
sequence = sequence[:n, ...]

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
                best_ref = (best_ref_id, x_cent, y_cent, seg)
        matches.add(best_ref)
    
    return matches

st1 = perf_counter()
for ix in range(0,n,1):
    s = perf_counter()
    seq = sequence[ix]
    prev = [None] * 5
    cache = {}
    for j in range(0,20,1):
        print('frame', j, flush=True)
        if (j < 4):
            continue
        
        ref_ids_list = []
        ref_ids_len_list = []
        for k in [4, 3, 2, 1, 0]:
            frame_idx = j - k
            print('frame look back', frame_idx, flush=True)
            if frame_idx not in cache:
                cache[frame_idx] = patoms(seq[frame_idx])
            seq_out_patoms = cache[frame_idx]
            best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare)
            
            ref_ids = np.array(sorted([i[0] for i in best_matches]), dtype=np.float32).tobytes()
            group_dicts[4-k][ref_ids] += 0.0000001
            ref_ids_list.append(ref_ids)
            ref_ids_len_list.append(len(ref_ids))

            if prev[4-k] is not None:
                cross = [i for i in product(prev[4-k], best_matches)]
                
                matches = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross]
                for i in matches:
                    vrlps[4-k][i] += 0.0000001
                direction = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross]
                magnitude = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], 
                                        dtype=np.float32).tobytes() for i in cross]
                vectors = [a + b + c for a, b, c in zip(matches, direction, magnitude)]
                for i in vectors:
                    vrlvs[4-k][i] += 0.0000001
            
            prev[4-k] = best_matches

        sequence_id = b"".join(ref_ids_list)
        sequence_dict[sequence_id] += 0.0000001
        ## separate dictionary to hold the lengths of each of the ref group ids
        sequence_dict_len[sequence_id] = ref_ids_len_list    
    
    e = perf_counter()
    print('seq_num:', ix, 'time taken (mins):', round((e-s)/60,4) )

en1 = perf_counter()
print('Time taken for 25 seqs (mins):', round((en1-st1)/60,4))

# write intelligence to disk
for i in range(len(vrlps)):
    with open(f'vrlp{i}.pkl', 'wb') as f:
        pickle.dump(vrlps[i], f)

for i in range(len(vrlvs)):
    with open(f'vrlv{i}.pkl', 'wb') as f:
        pickle.dump(vrlvs[i], f)

for i in range(len(group_dicts)):
    with open(f'group_dict{i}.pkl', 'wb') as f:
        pickle.dump(group_dicts[i], f)

with open('sequence_dict.pkl', 'wb') as f:
    pickle.dump(sequence_dict, f)

with open('sequence_dict_len.pkl', 'wb') as f:
    pickle.dump(sequence_dict_len, f)
