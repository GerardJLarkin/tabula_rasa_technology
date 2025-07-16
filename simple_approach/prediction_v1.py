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

from tabula_rasa_technology.simple_approach.compare_v1 import compare
from tabula_rasa_technology.simple_approach.patoms_v1 import patoms

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

## visual reference linking patoms
def find_best_matches(
    arrays: List[np.ndarray],
    references: List[np.ndarray],
    compare_func: Callable[[np.ndarray, np.ndarray], Tuple[str, str, float]]) -> Set[Tuple[str, float, float, float]]:
    
    matches: Set[Tuple[str, float, float, float]] = set()
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

start_prediction = []
st1 = perf_counter()
# this loop only looks at a single sequence set imported above
for ix in range(0,1,1):
    s = perf_counter()
    seq = sequence[ix]
    num_frames = 10 # this is the frame after which we will predict the next 4 frames (fingers crossed)
    working_memory_seq = 0
    for j in range(0,20,1):
        frame = seq[j]
        seq_out_patoms = patoms(frame)
        best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare)
        if num_frames == j:
            # print('work mem', working_memory_seq, 'seq num', j)
            # print('best matches', best_matches)
            
            # call the working memory frame relevant vrlp and vrlv files
            # given that we have the left hand side of the vrlv key pattern (second half of key), get all matching right hand side
            vrlp = vrlps[working_memory_seq] 
            vrlv = vrlvs[working_memory_seq] 
            for i in best_matches:
                # get all keys from vrlp where ref id in the best match is the first half of the key id-id pair, and has the highest value
                next_sequence_patom = {k: v for k, v in vrlp.items() if k.startswith(str(i[0]))}
                next_sequence_patom = list(sorted(next_sequence_patom.items(), key=lambda v: v[1], reverse=True))[0]
                #print(next_sequence_patom)
                
                # get all keys from vrlv where ref id in the best match is the start of the key, with the highest value
                next_sequence_vector = {k: v for k, v in vrlv.items() if k.startswith(str(i[0]))}
                next_sequence_vector = list(sorted(next_sequence_vector.items(), key=lambda v: v[1], reverse=True))[0]
                #print(next_sequence_vector)
                
                # is this the correct data to use to build the next predictive sequence?
                data_to_build_from = [working_memory_seq, next_sequence_patom, next_sequence_vector]
                start_prediction.append(data_to_build_from)

        working_memory_seq = (working_memory_seq + 1) % 5

print(start_prediction)

## need to predict next in sequence from start sequence data
## need to build each of the frames that created from the ref patoms in each of 
## the predicted frames, starting with the start sequence data
next_prediction = None
for i in range(1, 5):
    for j in start_prediction[1]:
        next_sequence_patom = {k:v for k, v in vrlp.items() if k.startswith(j[0])}


    #print(final_frame_compare_output)
    # create vectors
    # get next set of patoms
#     curr_seq_ref_patoms = [i[0] for i in final_frame_compare_output]
#     #print('curr', curr_seq_ref_patoms)
#     # find all next sequence ref patoms based on curr sequence ref patoms
#     next_seq_ref_patoms = []
#     for i in curr_seq_ref_patoms:
#         next_seq_keys = [(i, k, v) for k, v in vrlp.items() if k[:6] == i]
#         next_seq_key = max(next_seq_keys, key=lambda i:i[2])
#         next_seq_ref_patoms.append(next_seq_key[1][-6:])
#     #print('next',next_seq_ref_patoms)
#     # rebuld array to print as image
#     # need translation vector
#     # need centroid from set of patoms from last input frame
#     next_ref_patoms = []
#     for ix, i in enumerate(next_seq_ref_patoms):
#         next_patom = dict_ref_patom[i]
#         #print('nxtp', next_patom)
#         # reverse normalisation to find correct x, y positions
#         # print(next_patom[:,2] * (64 / 2))
#         x_vals = next_patom[:,1] * (64 / 2)
#         pseudo_orig_x_vals = final_frame_compare_output[ix][3] + x_vals
#         print('pseudo x vals',pseudo_orig_x_vals)
#         pseudo_orig_x_vals = pseudo_orig_x_vals.astype('int64').reshape(next_patom.shape[0],1)
#         y_vals = next_patom[:,2] * (64 / 2)
#         pseudo_orig_y_vals = final_frame_compare_output[ix][4] + y_vals
#         print('pseudo y vals',pseudo_orig_y_vals)
#         pseudo_orig_y_vals = pseudo_orig_y_vals.astype('int64').reshape(next_patom.shape[0],1)
#         pseudo_orig_array = np.hstack((pseudo_orig_x_vals, pseudo_orig_y_vals, next_patom[:,3].reshape(next_patom.shape[0],1)))
#         print('pseudo',pseudo_orig_array)
#         print(pseudo_orig_array.max())
#         next_ref_patoms.append(pseudo_orig_array)

#     array_image = []

# def merge_point_lists(
#     point_lists: List[np.ndarray],
#     shape: tuple[int, int],
#     mode: str = "sum"  # or "overwrite"
# ) -> np.ndarray:
#     """
#     Given a list of N×3 arrays, each with columns [x, y, value],
#     returns a single 2D array of shape `shape` filled with the
#     values from all lists at their (x,y) coords.

#     If mode=="sum", overlapping indices are summed;
#     if mode=="overwrite", later lists simply overwrite earlier.
#     """
#     result = np.zeros(shape, dtype=float)

#     for pts in point_lists:
#         # pts[:,0] = x indices, pts[:,1] = y indices, pts[:,2] = values
#         xs = pts[:, 0].astype(int)
#         ys = pts[:, 1].astype(int)
#         vals = pts[:, 2]

#         if mode == "sum":
#             # accumulate (works even if xs/ys contain duplicates)
#             np.add.at(result, (ys, xs), vals)
#         else:  # overwrite
#             result[ys, xs] = vals

#     return result

# shape = (64,64)
# result = merge_point_lists(next_ref_patoms, shape, mode="overwrite")

# print(result)

en1 = perf_counter()
print('Time taken for 100 seqs (mins):', round((en1-st1)/60,4))






# cnt = 0
# while cnt < 100:
#     for key, value in vrlv.items():
#         if value > 0:
#             print(key, value)
#             cnt += 1
