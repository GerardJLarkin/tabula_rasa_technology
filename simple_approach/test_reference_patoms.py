## add ignore warnings for now, will remove and debug once full algorithm is complete
# import warnings
# warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter
import numpy as np
import sys
import os

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.patoms import patoms
from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.unnormalise import unnormalise_xy

# --- load reference patoms once ---
root = os.path.dirname(os.path.abspath(__file__))
reference_dir = os.path.join(root, 'reference_patoms')
ref_patoms = [np.load(os.path.join(reference_dir, fn)) for fn in os.listdir(reference_dir)]


def best_matches_for_frame(frame):
    arrs = patoms(frame)
    matches = list()
    for arr in arrs:
        best_score = float('inf')
        best_tuple = None

        for ref in ref_patoms:
            id1, ref_id, score = compare(arr, ref)
            if (score < best_score) and (score < 1.5): 
                best_score = score
                # extract features once
                x_cent = float(arr[0, 1])
                y_cent = float(arr[0, 2])
                seg = float(arr[0, 3])
                min_x = float(arr[1, 0]) 
                max_x = float(arr[1, 1])
                min_y = float(arr[1, 2])
                max_y = float(arr[1, 3])
                best_tuple = (float(ref_id), x_cent, y_cent, best_score, min_x, max_x, min_y, max_y)

        if best_tuple is not None:
            matches.append(best_tuple)

    return matches, arrs

# def all_matches_for_frame(frame):
#
#     arrs = patoms(frame)
#     matches = list()
#     for arr in arrs:
#         for ref in REF_PATOMS:
#             pat_id, ref_id, score = compare(arr, ref) # switch back to using compare?
#             # extract features once
#             x_cent = float(arr[0, 1])
#             y_cent = float(arr[0, 2])
#             seg    = float(arr[0, 3])
#             best_tuple = (float(pat_id), float(ref_id), score)
#             matches.append(best_tuple)
#
#     return matches, arrs

# number of sequences to import
n = 1
# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[9325:9326, ...]
# print('loaded')

original_frame = dataset[0][10]

results = best_matches_for_frame(original_frame)
best_ref_matches = results[0]
original_patoms = results[1]
print('num ref patoms:', len(best_ref_matches))
print('num orig patoms:', len(original_patoms))
# print('best match ref', best_ref_matches[0])
# print('orig patom', original_patoms[0].shape)
# for i in best_ref_matches:
#     print(i)

# import matplotlib.pyplot as plt
# for ix, i in enumerate(best_ref_matches):
#     # read in ref_patoms
#     ref_patom = np.load(os.path.join(reference_dir, f'patom_{i[0]:.8f}.npy'))[2:,:3]
#     orig_patom = original_patoms[ix][2:,:3]
#     print('ref_patom\n', ref_patom)
#     print('orig_patom\n', orig_patom)
#     images = [orig_patom, ref_patom]
#     vmin = min([np.nanmin(i) for i in images])
#     vmax = max([np.nanmax(i) for i in images])
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
#     cmap = 'gray'
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
#     im = None
#     for ax, img in zip(axes, images):
#         im = ax.imshow(img, cmap=cmap, norm=norm)
#         ax.axis('off')

#     fig.suptitle(f'Orig vs Ref Score: {i[3]:.6f}', fontsize=20)
#     plt.show()

# from collections import defaultdict
# def group_by_first(sublists):
#     g = defaultdict(list)
#     for sub in sublists:
#         key = sub[0]
#         val = sub[1:]
#         g[key].append(val)
    
#     best_matches = []
#     best_key, best_val = min(g.items(), key=lambda kv: kv[1][1])
#     print(best_key, best_val)
#     best_matches.append((best_key, best_val))

#     return best_matches

# all_results = all_matches_for_frame(original_frame)

# all_matches = all_results[0]
# sorted_all_matches = sorted(all_matches, key=lambda x: (x[0], x[-1]))

# groups = group_by_first(sorted_all_matches)
# print(groups)
# orig_row_sum = []
# for i in original_patoms:
#     rows = i.shape[0]
#     orig_row_sum.append(rows)

# orig_num_rows = sum(orig_row_sum)
# print('orig rows', orig_num_rows)

# actual best match ref patoms
ref_patoms_list = []
for i in best_ref_matches:
    #print('orig patom', i[4],i[5],i[6],i[7])
    ref_patom = np.load(os.path.join(reference_dir, f'patom_{i[0]:.8f}.npy'))
    ref_patom[0,[1,2]] = [i[1],i[2]]
    #print('ref patom', ref_patom[1,:])
    ref_patom[1,:] = [i[4],i[5],i[6],i[7]]
    ref_patoms_list.append(ref_patom)

# ref_row_sum = []
# for i in ref_patoms_list:
#     rows = i.shape[0]
#     ref_row_sum.append(rows)

# ref_num_rows = sum(ref_row_sum)
# print('ref rows', ref_num_rows)

# choose image shape (must be large enough to include all x,y)
shape = (64, 64)

unnorm_list = []
for i in ref_patoms_list:
    #print(i.shape)
    unnorm = unnormalise_xy(i)
    #print(unnorm.shape)
    #print('norm', i[2:4,:], 'unnorm', unnorm[2:4,:])
    error = np.mean(i[2:,:2] != unnorm[2:,4:])
    #print('val error', error)
    diff_mask = i[2:,:2] != unnorm[2:,4:]
    diff_indices = np.argwhere(diff_mask)
    for idx in diff_indices:
        val_a = i[2:,:2][tuple(idx)]
        val_b = unnorm[2:,-2:][tuple(idx)]
        diff = val_a - val_b
        print(f"Index {tuple(idx)}: A={val_a}, B={val_b}, Diff={diff}")
    unnorm_list.append(unnorm)

# check for duplicates in the output of the list of patoms
patoms_stacked = np.vstack([i[2:,:3] for i in ref_patoms_list])
print('stacked patoms before de-dup', patoms_stacked.shape) # re collected patoms do not add back up to original frame size, why?
# remove duplicates - why do I have a lot of duplicates?
patoms_stacked = np.unique(patoms_stacked, axis=0)
print('stacked patoms after de-dup',patoms_stacked.shape) # re collected patoms do not add back up to original frame size, why?

# get unique x/y values in patoms stacked
vals = patoms_stacked[:,2]; print('patom unq vals', len(np.unique(vals).tolist()))

# check for duplicates in the output of the list of un-normalised patoms
unnorm_stacked = np.vstack([i[2:,:3] for i in unnorm_list])
print('stacked unnorm before de-dup', unnorm_stacked.shape) # re collected patoms do not add back up to original frame size, why?
# remove duplicates
unnorm_stacked = np.unique(unnorm_stacked, axis=0)
print('stacked unnorm after de-dup',unnorm_stacked.shape) # re collected patoms do not add back up to original frame size, why?


# recombine stacked patoms to check against original frame
h, w = shape

y = unnorm_stacked[:, 0].astype(np.int64, copy=False);y_list = np.unique(y).tolist() #; print('ref patom distinct y vals', (y_list))
x = unnorm_stacked[:, 1].astype(np.int64, copy=False);x_list = np.unique(x).tolist() #; print('ref patom distinct x vals', (x_list))
v = unnorm_stacked[:, 2].astype(np.int64, copy=False);v_list = np.unique(v).tolist() #; print('ref patom distinct v vals', (v_list))

# print('y shape', y.shape)
# print('x shape', x.shape)
# print('v shape', v.shape)

fill_value = np.mean(original_frame)

out = np.full(shape, 0, dtype=int) # cheap trick to make the images more alike by filling in with most common colour
# print('out shape', out.shape)
out[y, x] = v  # last occurrence wins due to NumPy assignment semantics
# print((out.shape))

# unique_out = np.unique(out).tolist()
# print(unique_out)

# accumulate sums and counts per pixel, then divide
# sums = np.zeros(shape, dtype=np.float64)
# cnts = np.zeros(shape, dtype=np.int64)
# np.add.at(sums, (y, x), v.astype(np.float64, copy=False))
# np.add.at(cnts, (y, x), 1)
# out = np.full(shape, fill_value, dtype=np.result_type(v.dtype, np.float32))
# nz = cnts > 0
# out[nz] = sums[nz] / cnts[nz]

import numpy as np
import matplotlib.pyplot as plt

img_patom = out
img_orig = original_frame

cmap = 'gray'

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

im = None
for ax, img in zip(axes, (img_orig, img_patom)):
    im = ax.imshow(img, cmap=cmap)
    ax.axis('off')

fig.suptitle('Original Frame vs Recreated Frame from Ref Patoms', fontsize=15)
plt.show()
#plt.imsave('ref_test.png', im, cmap='gray')


# print(out[:5,:5])
# print(original_frame[:5,:5])
orig_frame_patoms_match = np.mean(out != original_frame)
print('perc diff:', orig_frame_patoms_match)
diff_mask = out != original_frame
diff_indices = np.argwhere(diff_mask)
for idx in diff_indices:
    val_a = out[tuple(idx)]
    val_b = original_frame[tuple(idx)]
    diff = val_a - val_b
    # print(f"Index {tuple(idx)}: A={val_a}, B={val_b}, Diff={diff}")

matches = np.sum(out == original_frame)
total_elements = out.size
print('perc match:', matches/total_elements)
end = perf_counter()
print("Time taken (mins):", (end - start)/60)