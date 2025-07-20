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

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.patoms import patoms

root = os.path.dirname(os.path.abspath(__file__))

# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
# sequence = sequence[:100, ...]
# pick 1 sequence from within the training data to assess prediction algorithm
sequence = sequence[49:50, ...]
#print('loaded original data')
# load reference patoms
reference = os.path.join(root, 'reference_patoms')
dict_ref_patom = dict()
ref_patoms = [np.load(os.path.join(reference, fname), allow_pickle=True) for fname in os.listdir(reference)]
ref_ids = [i[0,0] for i in ref_patoms]
dict_ref_patom = dict.fromkeys(ref_ids, ref_patoms) 
#print('loaded reference patoms', len(ref_patoms))

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

## load group ids dictionaries
with open(root+'/group_dict0.pkl', 'rb') as g:
    group_dict0 = pickle.load(g)
with open(root+'/group_dict1.pkl', 'rb') as g:
    group_dict1 = pickle.load(g)
with open(root+'/group_dict2.pkl', 'rb') as g:
    group_dict2 = pickle.load(g)
with open(root+'/group_dict3.pkl', 'rb') as g:
    group_dict3 = pickle.load(g)
with open(root+'/group_dict4.pkl', 'rb') as g:
    group_dict4 = pickle.load(g)

grps = [group_dict0, group_dict1, group_dict2, group_dict3, group_dict4]

## load sequence dictionary
with open(root+'/sequence_dict.pkl', 'rb') as s:
    seq_dict = pickle.load(s)

## load sequence length dictionary
with open(root+'/sequence_dict_len.pkl', 'rb') as l:
    len_dict = pickle.load(l)


# seq_dict_test = dict(islice(seq_dict.items(), 1))

# for key, value in seq_dict_test.items():
#     lengths = len_dict[key]       # [n0,n1,n2,n3,n4]
#     # Decode the five blobs
#     blobs = []
#     offset = 0
#     for count in lengths:
#         print('dict byte cnt',count)
#         nbytes = count # float32 ⇒ 4 bytes each
#         print('nbytes', nbytes)          
#         chunk = key[offset : offset + nbytes]
#         print('chunk len', len(chunk))
#         blobs.append(chunk)
#         print(offset)
#         offset += nbytes
#     print('len dict',lengths)
#     print('blob len', [len(i) for i in blobs])


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

## function to compare the byte objects that represent the ref group ids
def similarity_ratio(a: bytes, b: bytes) -> float:
    matches = sum(x == y for x, y in product(a, b))
    return matches / (len(a) * len(b))

## find sequence that has the highest match against the 2 new ref group ids that are generated from fresh data
## ensures that when this function is used iteratively that any new subsets of ref group id pairs are identified
## they come from a point in the sequence that is less than or equal to the current current sequence index (forward recollection)
def find_best_sequence(
    d: Dict[bytes, float],
    length_map: Dict[bytes, List[int]],
    new_blob_A: bytes,
    new_blob_B: bytes,
    seq_ind: int = 0    
) -> Tuple[bytes, int, float, float]: #
    """
    Returns a tuple (best_key, best_pair_index, best_similarity, best_value)
    
    - best_key: the 5-blob bytes key in which we found the best-matching pair  
    - best_pair_index: i such that the match was against [blob_i, blob_{i+1}]  
    - best_similarity: the average of the two ordered similarity ratios  
    - best_value: d[best_key], used to break ties
    """
    best = None  # will hold (sim, value, key, index)
    
    for key, value in d.items():
        lengths = length_map[key]
        # Decode the five blobs
        blobs = []
        offset = 0
        for count in lengths:
            chunk = key[offset : offset + count]
            blobs.append(chunk)
            offset += count

        # Slide over 3 of the 4 consecutive 2-blob windows (potentially ignoring the last frame in the sequence - why do i have to do this?)
        for i in range(len(blobs) - 2):
            blob_i   = blobs[i]
            blob_i1  = blobs[i+1]
            # ordered similarity
            sim1 = similarity_ratio(new_blob_A, blob_i)
            sim2 = similarity_ratio(new_blob_B, blob_i1)
            avg_sim = (sim1 + sim2) / 2.0
            
            candidate = (avg_sim, value, key, i+2)
            # keep the largest (sim first, then value)
            if (best is None) or ((candidate[:2] > best[:2]) and (seq_ind <= best[3])):
                best = candidate
    
    best_sim, best_val, best_key, best_index = best
    return best_key, best_index, best_sim, best_val


## set next frame identification variables
num_frames = 7

## start prediction
st1 = perf_counter()
prev_best_matches = None
prev_ref_group_ids = []

for j in range(20):
    if j == num_frames:
        break
    else:
        frame = sequence[0][j]
        seq_out_patoms = patoms(frame)
        best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare)
        
        current_ref_group_ids = np.array(sorted([i[0] for i in best_matches]), dtype=np.float32).tobytes() # consider how to recover the ids?
        
        prev_ref_group_ids.append(current_ref_group_ids)
        
        if prev_best_matches is not None:
            cross = [i for i in product(prev_best_matches, best_matches)]
            matches = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross] # length of each byte object is 2
            direction = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross] # length of each byte object is 2
            magnitude = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross] # length of each byte object is 1
            vectors = [a + b + c for a, b, c in zip(matches, direction, magnitude)]
            
            # lengths = [2, 2, 1]
            # series_list = []
            #     offset = 0
            #     for count in lengths:
            #         n_bytes = count * 4
            #         chunk = i[offset:offset + n_bytes]
            #         labels = np.frombuffer(chunk, dtype=np.float32)
            #         series_list.append(labels)
            #         offset += n_bytes
            # print('ser_list',series_list[0:3])
            # print('exact match', vec_test[0][0] == series_list[0][0])
        
        prev_best_matches = best_matches


## new attempt to loop through and predict frames
max_frames_to_predict = 5
frame_number = 0
predicted_frames = []

while frame_number < max_frames_to_predict:
    best_sequence = find_best_sequence(seq_dict, len_dict, prev_ref_group_ids[-2], prev_ref_group_ids[-1])
    lengths = len_dict[best_sequence[0]]
    predicted_seq_index = best_sequence[1]
    sequence_value = best_sequence[3]
    # split best matched sequence to get the ref group ids at the index after the indexes identified in the best matched sequence
    seq_blobs = []
    offset = 0
    for count in lengths:
        chunk = best_sequence[0][offset : offset + count]
        seq_blobs.append(chunk)
        offset += count
    ## this line here is the current 'predicted' frame
    current_seq_ref_group_ids = seq_blobs[predicted_seq_index]
    prev_ref_group_ids.append(current_seq_ref_group_ids)
    
    prev_sequence_value = None
    ## if its the first loop iteration create the next sequence based on the first iteration
    if prev_sequence_value is None:
        next_best_sequence = find_best_sequence(seq_dict, len_dict, prev_ref_group_ids[-2], prev_ref_group_ids[-1], predicted_seq_index)
        next_lengths = len_dict[next_best_sequence[0]]
        next_predicted_seq_index = next_best_sequence[1]
        next_sequence_value = next_best_sequence[3]
        # split best matched sequence to get the ref group ids at the index after the indexes identified in the best matched sequence
        next_seq_blobs = []
        next_offset = 0
        for next_count in next_lengths:
            next_chunk = next_best_sequence[0][next_offset : next_offset + next_count]
            next_seq_blobs.append(next_chunk)
            next_offset += next_count
        ## this line here is the current 'predicted' frame
        next_current_seq_ref_group_ids = next_seq_blobs[next_predicted_seq_index]
        prev_ref_group_ids.append(next_current_seq_ref_group_ids)

        prev_sequence_value = next_sequence_value

    ## compare the current sequence value with the previous sequence value
    if prev_sequence_value <= sequence_value:
        # get next frame in original sequence
        current_seq_ref_group_ids = seq_blobs[predicted_seq_index + 1]
        predicted_frames.append(current_seq_ref_group_ids)
    else:
        predicted_frames.append(next_current_seq_ref_group_ids)
        
    ## if its the first loop iteration add the predicted frame
    if prev_sequence_value is None:
        predicted_frames.append(current_seq_ref_group_ids)

    prev_sequence_value = sequence_value
    frame_number += 1


print(len(predicted_frames))

en1 = perf_counter()
print('Time taken to predict 5 frames (mins):', round((en1-st1)/60,4))
