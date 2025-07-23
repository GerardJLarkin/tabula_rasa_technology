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


## visual reference linking patoms
def find_best_matches(
    arrays: List[np.ndarray],
    references: List[np.ndarray],
    compare_func: Callable[[np.ndarray, np.ndarray], Tuple[float, float, float]]
    ) -> Set[Tuple[float, float, float, float]]:
    
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
        
        current_ref_group_ids = np.array(sorted([i[0] for i in best_matches]), dtype=np.float32).tobytes()
        
        prev_ref_group_ids.append(current_ref_group_ids)
        
        if prev_best_matches is not None:
            cross = [i for i in product(prev_best_matches, best_matches)]
            matches = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross] 
            direction = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross]
            magnitude = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross]
            vectors = [a + b + c for a, b, c in zip(matches, direction, magnitude)]
        
        prev_best_matches = best_matches


### ok to here ###

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
) -> Tuple[bytes, int, float, float]: #
    """
    Returns a tuple (key (sequence id) of predicted frame, value of key of predicted frame, predicted_frame, index in sequence where predicted frame was found)
    
    - best_key: the 5-blob bytes key in which we found the best-matching pair  
    - best_pair_index: i such that the match was against [blob_i, blob_{i+1}]  
    - best_similarity: the average of the two ordered similarity ratios  
    - best_value: d[best_key], used to break ties
    """
    best = None  # will hold (sim, value, key, index)
    
    # for each key in the sequence dictionary, split it into its individual reference group ids
    for key, value in d.items():
        lengths = length_map[key]
        blobs = []
        offset = 0
        for count in lengths:
            chunk = key[offset : offset + count]
            blobs.append(chunk)
            offset += count
        
        # compare 2 pair of sequential ref grp id sliding windows against the 2 sequential ref grp ids initially submitted to function 
        # Slide over 3 of the 4 consecutive 2-blob windows (potentially ignoring the last frame in the sequence - why do i have to do this?)
        #  (0,1) (1,2) (2,3) (3,4)
        for i in range(3):
            blob_i = blobs[i]
            blob_i1 = blobs[i+1]
            # ordered similarity
            sim1 = similarity_ratio(new_blob_A, blob_i)
            sim2 = similarity_ratio(new_blob_B, blob_i1)
            avg_sim = (sim1 + sim2) / 2.0
            
            candidate = (avg_sim, value, key, i+2, blobs)
            # find the 2 ref group id sequential pair windows that are a closest match to the initially submitted ref group ids
            # keep the largest (sim first, then value)
            if (best is None) or (candidate[:2] > best[:2]):
                best = candidate
        
    best_sim, best_val, best_key, best_index, best_blobs = best
    
    ## extract predicted frame from best identified sequence
    predicted_ref_grp_ids = best_blobs[best_index]

    return best_key, best_val, predicted_ref_grp_ids, best_index


## new attempt to loop through and predict frames
max_frames_to_predict = 5
predicted_frames = [] # bytes objects
predicted_frames_seq_vals = []
ref_grp_id1, ref_gr_id2, index = prev_ref_group_ids[-2], prev_ref_group_ids[-1], 0

# find matching reference sequence once and then use existing dictionaries to get next frames
closest_matching_key, closest_matching_key_value, next_successive_frame_in_key, index_where_next_successive_frame_found \
    = find_best_sequence(seq_dict, len_dict, ref_grp_id1, ref_gr_id2)

predicted_frames.append(next_successive_frame_in_key)
predicted_frames_seq_vals.append(closest_matching_key_value)

# get sequence and sequence length dictionaries, find sequence where previously predicted frame is at index zero
# create dictionary of the original key with the first frame and its value 

for i in range(5):
    sequence_first_frame = defaultdict(list)
    sequences = []
    for key, value in seq_dict.items():
        lengths = len_dict[key]
        frames = []
        offset = 0
        for count in lengths:
            chunk = key[offset : offset + count]
            frames.append(chunk)
            offset += count
        sequence_first_frame[key] += [frames[0], value]

    # find sequence where the first frame is the same as the previously predicted frame
    matching_first_frames = {k: v[1] for k, v in sequence_first_frame.items() if v[0] == predicted_frames[-1]}
    # get the sequence that has the highest value (need a tie breaker)
    max_matching_frst_frame_key= max(matching_first_frames, key=matching_first_frames.get)
    # get next frame second position in matching key
    lengths = len_dict[max_matching_frst_frame_key]
    frames = []
    offset = 0
    for count in lengths:
        chunk = max_matching_frst_frame_key[offset : offset + count]
        frames.append(chunk)
        offest += count
    next_predicted_frame = frames[1]
    predicted_frames.append(next_predicted_frame)



print(len(predicted_frames))

en1 = perf_counter()
print('Time taken to predict 5 frames (mins):', round((en1-st1)/60,4))


#######################################################################################
################## rough work #########################################################
#######################################################################################
