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
from tabula_rasa_technology.simple_approach.unnormalise import unnormalise_xy

root = os.path.dirname(os.path.abspath(__file__))

# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
# sequence = sequence[:100, ...]
# pick 1 sequence from within the training data to assess prediction algorithm
sequence = sequence[12:13, ...]
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

#     offset = 0
#     for count in lengths:
#         n_bytes = count * 4
#         chunk = i[offset:offset + n_bytes]
#         labels = np.frombuffer(chunk, dtype=np.float32)
#         series_list.append(labels)
#         offset += n_bytes
# print('ser_list',series_list[0:3])
# print('exact match', vec_test[0][0] == series_list[0][0])

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

## set next frame identification variables
num_frames = 2


# for j in range(20):
#     if j == num_frames:
#         break
#     else:
#         frame = sequence[0][j]
#         seq_out_patoms = patoms(frame)
#         best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare)
        
#         current_ref_group_ids = np.array(sorted([i[0] for i in best_matches]), dtype=np.float32).tobytes()
        
#         prev_ref_group_ids.append(current_ref_group_ids)
        
#         if prev_best_matches is not None:
#             cross = [i for i in product(prev_best_matches, best_matches)]
#             matches = [np.array([i[0][0], i[1][0]], dtype=np.float32).tobytes() for i in cross] 
#             direction = [np.array([i[0][3], i[1][3]], dtype=np.float32).tobytes() for i in cross]
#             magnitude = [np.array([np.sqrt((i[0][1] - i[1][1])**2 + (i[0][2]-i[1][2])**2) / 89.1], dtype=np.float32).tobytes() for i in cross]
#             vectors = [a + b + c for a, b, c in zip(matches, direction, magnitude)]
        
#         prev_best_matches = best_matches


# ### ok to here ###

## start prediction
st1 = perf_counter()
prev_best_matches = None
prev_ref_group_ids = []

vrlp_dict = defaultdict(float)
vrlv_dict = defaultdict(float)
current_ref_ids = []
num_patoms_list = []
for j in range(20):
    dict_idx = j % 5
    if j == num_frames:
        frame = sequence[0][j]
        seq_out_patoms = patoms(frame)
        num_patoms = len(seq_out_patoms)
        num_patoms_list.append(num_patoms)
        best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare)
        
        for i in best_matches:
            ref_ids = str(f'{i[0]:.8f}')
            print(ref_ids)
        current_ref_ids.append(ref_ids)

        vrlp = vrlps[dict_idx]
        for k, v in vrlp.items():
            dict_keys = np.frombuffer(k, dtype=np.float32)
            keys = str(f'{dict_keys[0]:.8f}')+str(f'{dict_keys[1]:.8f}')
            
            vrlp_dict[keys] = v

        vrlv = vrlvs[dict_idx]
        for k, v in vrlv.items():
            dict_keys = np.frombuffer(k, dtype=np.float32)
            keys = str(f'{dict_keys[0]:.8f}')+str(f'{dict_keys[1]:.8f}')
            
            vrlv_dict[keys] = v
    else:
        continue


next_frame_keys = [(k[11:], v) for k, v in vrlp_dict.items() if k[:11] in current_ref_ids]   
print(next_frame_keys)

# targets = current_ref_ids
# best = {}
# for k, v in d.items():
#     if len(k) < 8: 
#         continue
#     p = k[:8]
#     if p in targets and (p not in best or v > best[p][1]):
#         best[p] = (k, v)
# return best

### ok to here ###

# ## function to compare the byte objects that represent the ref group ids
# def similarity_ratio(a: bytes, b: bytes) -> float:
#     matches = sum(x == y for x, y in product(a, b))
#     return matches / (len(a) * len(b))

# ## find sequence that has the highest match against the 2 new ref group ids that are generated from fresh data
# ## ensures that when this function is used iteratively that any new subsets of ref group id pairs are identified
# ## they come from a point in the sequence that is less than or equal to the current current sequence index (forward recollection)
# def find_best_sequence(d,length_map,new_blob_A,new_blob_B): #
#     """
#     Returns a tuple (key (sequence id) of predicted frame, value of key of predicted frame, predicted_frame, index in sequence where predicted frame was found)
    
#     - best_key: the 5-blob bytes key in which we found the best-matching pair  
#     - best_pair_index: i such that the match was against [blob_i, blob_{i+1}]  
#     - best_similarity: the average of the two ordered similarity ratios  
#     - best_value: d[best_key], used to break ties
#     """
#     best = None  # will hold (sim, value, key, index)
    
#     # for each key in the sequence dictionary, split it into its individual reference group ids
#     for key, value in d.items():
#         lengths = length_map[key]
#         blobs = []
#         offset = 0
#         for count in lengths:
#             chunk = key[offset : offset + count]
#             blobs.append(chunk)
#             offset += count
        
#         # compare 2 pair of sequential ref grp id sliding windows against the 2 sequential ref grp ids initially submitted to function 
#         # Slide over 3 of the 4 consecutive 2-blob windows (potentially ignoring the last frame in the sequence - why do i have to do this?)
#         #  (0,1) (1,2) (2,3) (3,4)
#         for i in range(3):
#             blob_i = blobs[i]
#             blob_i1 = blobs[i+1]
#             # ordered similarity
#             sim1 = similarity_ratio(new_blob_A, blob_i)
#             sim2 = similarity_ratio(new_blob_B, blob_i1)
#             avg_sim = (sim1 + sim2) / 2.0
            
#             candidate = (avg_sim, value, i+2, blobs)
#             # find the 2 ref group id sequential pair windows that are a closest match to the initially submitted ref group ids
#             # keep the largest (sim first, then value)
#             if (best is None) or (candidate[:2] > best[:2]):
#                 best = candidate
        
#     best_sim, best_val, best_index, best_blobs = best
    
#     ## extract predicted frame from best identified sequence
#     predicted_ref_grp_ids = best_blobs[best_index]

#     return predicted_ref_grp_ids


# ## new attempt to loop through and predict frames
# max_frames_to_predict = 5
# predicted_frames = [] # bytes objects
# ref_grp_id1, ref_gr_id2 = prev_ref_group_ids[-2], prev_ref_group_ids[-1]

# # find matching reference sequence once and then use existing dictionaries to get next frames
# inital_predicted_frame = find_best_sequence(seq_dict, len_dict, ref_grp_id1, ref_gr_id2)

# predicted_frames.append(inital_predicted_frame)

# print('len predicted frames',len(predicted_frames))

# # get sequence and sequence length dictionaries, find sequence where previously predicted frame is at index zero
# # create dictionary of the original key with the first frame and its value 

# for i in range(5):
#     sequence_first_frame = defaultdict(list)
#     sequences = []
#     for key, value in seq_dict.items():
#         lengths = len_dict[key]
#         frames = []
#         offset = 0
#         for count in lengths:
#             chunk = key[offset : offset + count]
#             frames.append(chunk)
#             offset += count
#         sequence_first_frame[key] += [frames[0], value]
        
#         print('pred frame', predicted_frames[-1]==frames[0])
    
#     # find sequence where the first frame is the same as the previously predicted frame
#     # matching_first_frames = {k: v[1] for k, v in sequence_first_frame.items() if v[0] == predicted_frames[-1]}
#     # # get the sequence that has the highest value (need a tie breaker)
#     # max_matching_first_frame_key= max(matching_first_frames, key=matching_first_frames.get)
#     # # get next frame second position in matching key
#     # lengths = len_dict[max_matching_first_frame_key]
#     # frames = []
#     # offset = 0
#     # for count in lengths:
#     #     chunk = max_matching_first_frame_key[offset : offset + count]
#     #     frames.append(chunk)
#     #     offest += count
#     # next_predicted_frame = frames[1]
#     # predicted_frames.append(next_predicted_frame)
#     # print('2nd len predicted frames', len(predicted_frames))



# print(len(predicted_frames))

en1 = perf_counter()
print('Time taken to predict 5 frames (mins):', round((en1-st1)/60,4))


#######################################################################################
################## rough work #########################################################
#######################################################################################
