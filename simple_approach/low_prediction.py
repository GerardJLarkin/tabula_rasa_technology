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

# folder paths
root = os.path.dirname(__file__)

# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
sequence = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
# sequence = sequence[:100, ...]
# pick 1 sequence from within the training data to assess prediction algorithm
n = 1
sequence = sequence[n:n+1, ...]
#print('loaded original data')
# load reference patoms
reference = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference, fname)) for fname in os.listdir(reference)]

## load visual reference linking patoms
with open(root+'/vrlp_dict.pkl', 'rb') as fp:
    vrlp = pickle.load(fp)
with open(root+'/vrlv_dict.pkl', 'rb') as fp:
    vrlv = pickle.load(fp)


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
                min_x = arr[1,0]
                max_x = arr[1,1]
                min_y = arr[1,2]
                max_y = arr[1,3]
                # 0, 1, 2, 3, 4, 5, 6, 7
                best_ref = (best_ref_id, seg, x_cent, y_cent, min_x, max_x, min_y, max_y)
        matches.add(best_ref)
    
    return matches

## set exit frame to allow prediction of next frame
num_frames = 2

## start prediction
st1 = perf_counter()

frame = sequence[0][3]
next_frame = sequence[0][4]

# 1 get base patoms
seq_out_patoms = patoms(frame)

# 2 find best matched ref patoms
best_matches = find_best_matches(seq_out_patoms, ref_patoms, compare) 

# 6 get the vrlv dictionary key that has the follow on key as the first part of its own key
in_scope_vrlv_keys_sublists = [[] for _ in range(len(best_matches))]
for idx, i in enumerate(best_matches):
    ref_match = np.array([i[0],i[1]], dtype=np.float32)
    for k, v in vrlv.items():
        dict_keys = np.frombuffer(k, dtype=np.float32)
        if np.array_equal(ref_match, dict_keys[[0,2]]):
            in_scope_vrlv_keys_sublists[idx].append((np.append(dict_keys,i[2:]), v))

# 7 get the dictionary key with the highest value
predict_keys = []
for idx, i in enumerate(in_scope_vrlv_keys_sublists):
    max_key_val_per_ref_id = sorted(i, key=lambda x: x[1], reverse=True)[0][0]
    predict_keys.append(max_key_val_per_ref_id)  

###############################################################
# reconstruct patoms from here on out with each predicted key #
###############################################################
shape = (64, 64)
predicted_ref_patoms = [i[1] for i in predict_keys]
predicted_frame_ref_patoms = [np.load(os.path.join(reference, f'patom_{i:.8f}.npy')) for i in predicted_ref_patoms]

adjusted_predicted_frame_ref_patoms = []
for idx, i in enumerate(predicted_frame_ref_patoms):
    centroid = predict_keys[idx][[5,6]]
    patom_bounds = predict_keys[idx][7:]
    #print(centroid); print(type(centroid))
    i[0,[1,2]] = centroid
    i[1,:] = patom_bounds
    adjusted_predicted_frame_ref_patoms.append(i)

unnorm_list = []
for i in adjusted_predicted_frame_ref_patoms:
    unnorm = unnormalise_xy(i)
    unnorm_list.append(unnorm)

# check for duplicates in the output of the list of un-normalised patoms
unnorm_stacked = np.vstack([i[2:,:3] for i in unnorm_list])
unnorm_stacked = np.unique(unnorm_stacked, axis=0)
h, w = shape

y = unnorm_stacked[:, 0].astype(np.int64, copy=False)
x = unnorm_stacked[:, 1].astype(np.int64, copy=False)
v = unnorm_stacked[:, 2]

out = np.full(shape, 0, dtype=int)
out[y, x] = v 

import numpy as np
import matplotlib.pyplot as plt

img_patom = out
img_orig = next_frame

images = [next_frame, out]

cmap = 'gray'

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

im = None
for ax, img in zip(axes, images):
    im = ax.imshow(img, cmap=cmap)
    ax.axis('off')

fig.suptitle('Original Next Frame vs Predicted Next Frame', fontsize=15)
plt.show()
plt.imsave('patom_test.png', img, cmap='gray')

# print(out[:5,:5])
# print(original_frame[:5,:5])
orig_frame_patoms_match = np.mean(out != next_frame)
print('% mismatch with zeros',orig_frame_patoms_match)
# remove zeros and calculate mistmatch
out_nonzero = out.copy().astype('float32')
out_nonzero[out_nonzero == 0] = float(np.nan)
orig_nonzero = next_frame.copy().astype('float32')
orig_nonzero[orig_nonzero == 0] = float(np.nan)

mask = ~np.isnan(out_nonzero) & ~np.isnan(orig_nonzero)
out_nonzero = out_nonzero[mask]
orig_nonzero = orig_nonzero[mask]

orig_frame_patoms_match_nonzero = np.mean(out_nonzero != orig_nonzero)
print('% mismatch without zeros',orig_frame_patoms_match_nonzero)
diff_mask = out != next_frame
diff_indices = np.argwhere(diff_mask)

# print('orig nonzero', orig_nonzero.shape)
# print('out nonzero', out_nonzero.shape)
diffs = []

for idx in diff_indices:
    val_a = out[tuple(idx)]
    val_b = next_frame[tuple(idx)]
    diff = val_a - val_b
    #print(f"Index {tuple(idx)}: A={val_a}, B={val_b}, Diff={diff}")
    diffs.append(diff)

print('% mismatch nonzero:', 1 - ((orig_nonzero.shape[-1] - len(diffs)) / orig_nonzero.shape[-1]) )

en1 = perf_counter()
print('Time taken to predict 1 frames (mins):', round((en1-st1)/60,4))
