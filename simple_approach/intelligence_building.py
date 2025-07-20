# intelligence building
import numpy as np
import os
import sys
from time import perf_counter
from typing import Callable, List, Tuple, Set
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
# load reference patoms
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]

## visual reference linking patoms
vrlp0 = defaultdict(float)
vrlp1 = defaultdict(float)
vrlp2 = defaultdict(float)
vrlp3 = defaultdict(float)
vrlp4 = defaultdict(float)

## visual reference linking vector
vrlv0 = defaultdict(float)
vrlv1 = defaultdict(float)
vrlv2 = defaultdict(float)
vrlv3 = defaultdict(float)
vrlv4 = defaultdict(float)

# instatiate group dictionary
group_dict0 = defaultdict(float)
group_dict1 = defaultdict(float)
group_dict2 = defaultdict(float)
group_dict3 = defaultdict(float)
group_dict4 = defaultdict(float)

# instatiate sequence dictionary
sequence_dict = defaultdict(float)

# instatiate sequence len (of each byte object) dictionary
sequence_dict_len = defaultdict(list)

# use the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# due to memory/processing limitations only use the first 100 of the 10000 total examples.
n = 2
sequence = sequence[:n, ...]

## visual reference linking patoms
def find_best_matches(
    arrays: List[np.ndarray],
    references: List[np.ndarray],
    compare_func: Callable[[np.ndarray, np.ndarray], Tuple[float, float, float]]) -> Set[Tuple[float, float, float, float]]:
    
    matches: Set[Tuple[float, float, float, float]] = set()
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

st1 = perf_counter()
for ix in range(0,n,1):
    s = perf_counter()
    seq = sequence[ix]
    
    prev0 = None; prev1 = None; prev2 = None; prev3 = None; prev4 = None
    for j in range(0,20,1):
        if (j < 4): # | (j > 20):
            continue
        else:
            frame0 = seq[j-4]
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

            frame1 = seq[j-3]
            seq_out_patoms1 = patoms(frame1)
            best_matches1 = find_best_matches(seq_out_patoms1, ref_patoms, compare)
            
            ref_ids1 = np.array(sorted([i[0] for i in best_matches1])).tobytes()
            group_dict1[ref_ids1] += 0.0000001
            
            if prev1 is not None:
                cross1 = [i for i in product(prev1, best_matches1)]
                matches1 = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross1]
                for i in matches1:
                    vrlp1[i] += 0.0000001
                direction1 = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross1]
                magnitude1 = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross1]
                vectors1 = [a + b + c for a, b, c in zip(matches1, direction1, magnitude1)]
                for i in vectors1:
                    vrlv1[i] += 0.0000001
            prev1 = best_matches1

            frame2 = seq[j-2]
            seq_out_patoms2 = patoms(frame2)
            best_matches2 = find_best_matches(seq_out_patoms2, ref_patoms, compare)
            
            ref_ids2 = np.array(sorted([i[0] for i in best_matches2])).tobytes()
            print('ref_ids2',len(ref_ids2))
            group_dict2[ref_ids2] += 0.0000001
            
            if prev2 is not None:
                cross2 = [i for i in product(prev2, best_matches2)]
                matches2 = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross2]
                for i in matches2:
                    vrlp2[i] += 0.0000001
                direction2 = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross2]
                magnitude2 = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross2]
                vectors2 = [a + b + c for a, b, c in zip(matches2, direction2, magnitude2)]
                for i in vectors2:
                    vrlv2[i] += 0.0000001
            prev2 = best_matches2

            frame3 = seq[j-1]
            seq_out_patoms3 = patoms(frame3)
            best_matches3 = find_best_matches(seq_out_patoms3, ref_patoms, compare)
            
            ref_ids3 = np.array(sorted([i[0] for i in best_matches3])).tobytes()
            print('ref_ids3',len(ref_ids3))
            group_dict3[ref_ids3] += 0.0000001
            
            if prev3 is not None:
                cross3 = [i for i in product(prev3, best_matches3)]
                matches3 = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross3]
                for i in matches3:
                    vrlp3[i] += 0.0000001
                direction3 = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross3]
                magnitude3 = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross3]
                vectors3 = [a + b + c for a, b, c in zip(matches3, direction3, magnitude3)]
                for i in vectors3:
                    vrlv3[i] += 0.0000001
            prev3 = best_matches3

            frame4 = seq[j]
            seq_out_patoms4 = patoms(frame4)
            best_matches4 = find_best_matches(seq_out_patoms4, ref_patoms, compare)
            
            ref_ids4 = np.array(sorted([i[0] for i in best_matches4])).tobytes()
            print('ref_ids4',len(ref_ids4))
            group_dict4[ref_ids4] += 0.0000001
            
            if prev4 is not None:
                cross4 = [i for i in product(prev4, best_matches4)]
                matches4 = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross4]
                for i in matches4:
                    vrlp4[i] += 0.0000001
                direction4 = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross4]
                magnitude4 = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross4]
                vectors4 = [a + b + c for a, b, c in zip(matches4, direction4, magnitude4)]
                for i in vectors4:
                    vrlv4[i] += 0.0000001
            prev4 = best_matches4
            
            sequence_id = ref_ids0 + ref_ids1 + ref_ids2 + ref_ids3 + ref_ids4
            sequence_dict[sequence_id] += 0.0000001
            ## separate dictionary to hold the lengths of each of the ref group ids
            sequence_dict_len[sequence_id] = [len(ref_ids0), len(ref_ids1), len(ref_ids2), len(ref_ids3), len(ref_ids4)]
    
    
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

print('group_dict0', sys.getsizeof(group_dict0))
with open('group_dict0.pkl', 'wb') as f:
    pickle.dump(group_dict0, f)
with open('group_dict1.pkl', 'wb') as f:
    pickle.dump(group_dict1, f)
with open('group_dict2.pkl', 'wb') as f:
    pickle.dump(group_dict2, f)
with open('group_dict3.pkl', 'wb') as f:
    pickle.dump(group_dict3, f)
with open('group_dict4.pkl', 'wb') as f:
    pickle.dump(group_dict4, f)

print('sequence_dict', sys.getsizeof(sequence_dict))
with open('sequence_dict.pkl', 'wb') as f:
    pickle.dump(sequence_dict, f)

print('sequence_dict_len', sys.getsizeof(sequence_dict_len))
with open('sequence_dict_len.pkl', 'wb') as f:
    pickle.dump(sequence_dict_len, f)