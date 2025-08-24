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

from tabula_rasa_technology.simple_approach.compare_v1 import compare
from tabula_rasa_technology.simple_approach.patoms_v1 import patoms

root = os.path.dirname(os.path.abspath(__file__))
# load reference patoms
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]

## visual reference linking patoms
# ref_ids = [str(i[0,0]) for i in ref_patoms]
# vrlp_keys = [i[0]+i[1] for i in product(ref_ids, repeat=2)]
# vrlp0 = dict.fromkeys(vrlp_keys, 0) # 61.5MB to start
# print('vrlp',sys.getsizeof(vrlp0))
# vrlp1 = dict.fromkeys(vrlp_keys, 0)
# vrlp2 = dict.fromkeys(vrlp_keys, 0)
# vrlp3 = dict.fromkeys(vrlp_keys, 0)
# vrlp4 = dict.fromkeys(vrlp_keys, 0)

vrlp0 = defaultdict(float)
vrlp1 = defaultdict(float)
vrlp2 = defaultdict(float)
vrlp3 = defaultdict(float)
vrlp4 = defaultdict(float)

## visual reference linking vector
# segments = [f"{i[0]:0>2}"+f"{i[1]:0>2}" for i in product((range(16)), repeat=2)] # 00 to 15
# magnitudes = ['00','01','02','03','04','05','06','07','08','09'] # 00 to 09
# vectors = [i[0]+i[1] for i in product(segments, magnitudes)]
# vrlv_keys = [i[0]+i[1] for i in product(ref_ids, vectors)]
# vrlv0 = dict.fromkeys(vrlv_keys, 0.0) # 1GB to start
# print('vrlv',sys.getsizeof(vrlv0))
# vrlv1 = dict.fromkeys(vrlv_keys, 0.0)
# vrlv2 = dict.fromkeys(vrlv_keys, 0.0)
# vrlv3 = dict.fromkeys(vrlv_keys, 0.0)
# vrlv4 = dict.fromkeys(vrlv_keys, 0.0)

vrlv0 = defaultdict(float)
vrlv1 = defaultdict(float)
vrlv2 = defaultdict(float)
vrlv3 = defaultdict(float)
vrlv4 = defaultdict(float)

# del objects to free up memory
#del ref_ids; del vrlp_keys; del segments; del magnitudes; del vectors; del vrlv_keys

# instatiate group dictionary
group_dict0 = defaultdict(float)
group_dict1 = defaultdict(float)
group_dict2 = defaultdict(float)
group_dict3 = defaultdict(float)
group_dict4 = defaultdict(float)

# instatiate sequence dictionary
sequence_dict = defaultdict(float)

# use the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# due to memory/processing limitations only use the first 100 of the 10000 total examples.
sequence = sequence[:50, ...]

## visual reference linking patoms
def find_best_matches(arrays, references, compare_func):
    
    matches = set()
    for arr in arrays:
        best_score = float('inf')
        best_ref_id: str = None
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
for ix in range(0,50,1):
    s = perf_counter()
    seq = sequence[ix]
    
    prev0 = None; prev1 = None; prev2 = None; prev3 = None; prev4 = None
    for j in range(0,20,1):
        if (j < 4) | (j > 20):
            continue
        else:
            frame0 = seq[j-4]
            seq_out_patoms0 = patoms(frame0)
            best_matches0 = find_best_matches(seq_out_patoms0, ref_patoms, compare)
            ref_ids0 = sorted([str(i[0]) for i in best_matches0])
            group0_id = ','.join(ref_ids0)
            group_dict0[group0_id] += 0.0000001
            if prev0 is not None:
                cross0 = [i for i in product(prev0, best_matches0)]
                matches0 = [str(i[0][0])+str(i[1][0]) for i in cross0]
                for i in matches0:
                    vrlp0[i] += 0.0000001
                direction0 = [f"{int(i[0][3]):0>2}"+f"{int(i[1][3]):0>2}" for i in cross0]
                magnitude0 = [f"{int(round((np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2)) / 89.1,1)*10):0>2}" for i in cross0]
                vectors0 = ['0.'+a.split('0.',2)[-1]+b+c for a, b, c in zip(matches0, direction0, magnitude0)]
                for i in vectors0:
                    vrlv0[i] += 0.0000001
            prev0 = best_matches0

            frame1 = seq[j-3]
            seq_out_patoms1 = patoms(frame1)
            best_matches1 = find_best_matches(seq_out_patoms1, ref_patoms, compare)
            ref_ids1 = sorted([str(i[0]) for i in best_matches1])
            group1_id = ','.join(ref_ids1)
            group_dict1[group1_id] += 0.0000001
            if prev1 is not None:
                cross1 = [i for i in product(prev1, best_matches1)]
                matches1 = [str(i[0][0])+str(i[1][0]) for i in cross1]
                for i in matches1:
                    vrlp1[i] += 0.0000001
                direction1 = [f"{int(i[0][3]):0>2}"+f"{int(i[1][3]):0>2}" for i in cross1]
                magnitude1 = [f"{int(round((np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2)) / 89.1,1)*10):0>2}" for i in cross1]
                vectors1 = ['0.'+a.split('0.',2)[-1]+b+c for a, b, c in zip(matches1, direction1, magnitude1)]
                for i in vectors1:
                    vrlv1[i] += 0.0000001
            prev1 = best_matches1

            frame2 = seq[j-2]
            seq_out_patoms2 = patoms(frame2)
            best_matches2 = find_best_matches(seq_out_patoms2, ref_patoms, compare)
            ref_ids2 = sorted([str(i[0]) for i in best_matches2])
            group2_id = ','.join(ref_ids2)
            group_dict2[group2_id] += 0.0000001
            if prev2 is not None:
                cross2 = [i for i in product(prev2, best_matches2)]
                matches2 = [str(i[0][0])+str(i[1][0]) for i in cross2]
                for i in matches2:
                    vrlp2[i] += 0.0000001
                direction2 = [f"{int(i[0][3]):0>2}"+f"{int(i[1][3]):0>2}" for i in cross2]
                magnitude2 = [f"{int(round((np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2)) / 89.1,1)*10):0>2}" for i in cross2]
                vectors2 = ['0.'+a.split('0.',2)[-1]+b+c for a, b, c in zip(matches2, direction2, magnitude2)]
                for i in vectors2:
                    vrlv2[i] += 0.0000001
            prev2 = best_matches2

            frame3 = seq[j-1]
            seq_out_patoms3 = patoms(frame3)
            best_matches3 = find_best_matches(seq_out_patoms3, ref_patoms, compare)
            ref_ids3 = sorted([str(i[0]) for i in best_matches3])
            group3_id = ','.join(ref_ids3)
            group_dict3[group3_id] += 0.0000001
            if prev3 is not None:
                cross3 = [i for i in product(prev3, best_matches3)]
                matches3 = [str(i[0][0])+str(i[1][0]) for i in cross3]
                for i in matches3:
                    vrlp3[i] += 0.0000001
                direction3 = [f"{int(i[0][3]):0>2}"+f"{int(i[1][3]):0>2}" for i in cross3]
                magnitude3 = [f"{int(round((np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2)) / 89.1,1)*10):0>2}" for i in cross3]
                vectors3 = ['0.'+a.split('0.',2)[-1]+b+c for a, b, c in zip(matches3, direction3, magnitude3)]
                for i in vectors3:
                    vrlv3[i] += 0.0000001
            prev3 = best_matches3

            frame4 = seq[j]
            seq_out_patoms4 = patoms(frame4)
            best_matches4 = find_best_matches(seq_out_patoms4, ref_patoms, compare)
            ref_ids4 = sorted([str(i[0]) for i in best_matches4])
            group4_id = ','.join(ref_ids4)
            group_dict4[group4_id] += 0.0000001
            if prev4 is not None:
                cross4 = [i for i in product(prev4, best_matches4)]
                matches4 = [str(i[0][0])+str(i[1][0]) for i in cross4]
                for i in matches4:
                    vrlp4[i] += 0.0000001
                direction4 = [f"{int(i[0][3]):0>2}"+f"{int(i[1][3]):0>2}" for i in cross4]
                magnitude4 = [f"{int(round((np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2)) / 89.1,1)*10):0>2}" 
                                        for i in cross4]
                vectors4 = ['0.'+a.split('0.',2)[-1]+b+c for a, b, c in zip(matches4, direction4, magnitude4)]
                for i in vectors4:
                    vrlv4[i] += 0.0000001
            prev4 = best_matches4
            
            sequence_id = '#0#' + group0_id + '#1#' + group1_id + '#2#' + group2_id + '#3#' + group3_id + '#4#' + group4_id
            sequence_dict[sequence_id] += 0.0000001
    
    
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