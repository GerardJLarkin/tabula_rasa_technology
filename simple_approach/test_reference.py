# function to create actual reference patom from group members
import numpy as np
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt
import glob, os
import sys

sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

# your imports
from tabula_rasa_technology.simple_approach.reference import create_reference_patom
from tabula_rasa_technology.simple_approach.compare import compare

historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(__file__)
historic = os.path.join(root, 'test_historic_data')
output = os.path.join(root, 'test_reference_patoms')
os.makedirs(output, exist_ok=True)

# function to create actual reference patom from group members
import numpy as np

def split_rows(arr, avg_rows):
 
    array_shape = arr.shape[0]
    if array_shape < avg_rows:
        print('avg rows', avg_rows)
        print('arr shape', array_shape)
    # id the shape of the indivdiaul patom is greater or equal to the average number of rows
    # split the array into subset arrays based on the number of rows 
    if (array_shape >= avg_rows):
        #print('avg rows', avg_rows)
        #print('arr shape', array_shape)
        # caluclate number of subset arrays
        subset = array_shape // avg_rows
        # get modulo of array shape and avg number of rows to determine in rows need to be allocated
        # to more than one subset
        mod  = array_shape % avg_rows  
        #print('mod', mod)
        # create array to allow subset end values to be determined (so we can split the patom array correctly based on row index)
        sizes = np.full(avg_rows, subset, dtype=int)
        #print('sizes', sizes)
        if mod != 0:
            print('****************************************************************')
            # find index locations to add an extra row to distribute the add'n rows evenly across the subsets
            extra_pos = np.floor(np.arange(mod) * avg_rows / mod).astype(int)
            print('extra pos', extra_pos)
            sizes[extra_pos] += 1
            print('sizes',sizes)
        #print('cumsum', np.cumsum(sizes))
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        #print('starts', starts)
        #print('ends', starts + sizes)
        
        return [arr[s:e] for s, e in zip(starts, starts + sizes)]

    # if the patom array has a smaller shape than the average rows, repeat rows evenly across the subsets
    idx = np.floor(np.arange(avg_rows) * array_shape / avg_rows).astype(int)
    #print('idx',idx)
    
    return [arr[i:i+1] for i in idx]

def create_reference_patom(arrays):

    # get second rows to find average min_x, max_x, min_y, max_y
    group_second_rows = np.vstack([pat[[1],:] for pat in arrays])
    group_second_rows = np.nanmean(group_second_rows, axis=0, keepdims=True)

    group_values = np.vstack([pat[2:,:3] for pat in arrays])
    avg_rows = int(np.ceil(group_values.shape[0] / len(arrays))) # is it better to have a higher or lower number of avg rows?
    
    # create an array for each row in the average number of rows
    # this is to hold row(s) corresponding to the split peformed on the patoms 
    split_arrays = [[] for _ in range(avg_rows)]
    
    # split each patom into subsections, then rejoin corresponding subsections to calulcate average values 
    # for each section and then stitch these back together to create a reference patom
    for i in arrays:
        arr = i[2:,:3]
        subarrays = split_rows(arr, avg_rows)
        # for each set of 'rows' in the subarray, sum along axis 0 and divide by the number of rows in the split
        for idx, j in enumerate(subarrays):
            first_sum = np.sum(j, axis=0, keepdims=True)
            split_arrays[idx].append(first_sum)

    
    stacked_split_arrays = []
    for row_array in split_arrays:
        stacked_rows = np.vstack((row_array))
        second_sum = np.sum(stacked_rows, axis=0, keepdims=True) / stacked_rows.shape[0]
        stacked_split_arrays.append(second_sum)
    
    reference_array_values = np.vstack((stacked_split_arrays))

    ref_patom_values = np.hstack((reference_array_values, np.full((reference_array_values.shape[0],1), np.nan)))

    ref_id = np.random.default_rng().random(dtype=np.float32)

    reference_patom = np.vstack((
        np.array([[ref_id, np.nan, np.nan, np.nan]]),
        group_second_rows, 
        ref_patom_values))

    return reference_patom

# load
with open("groups.pkl", "rb") as f:
    groups = pickle.load(f)

for group in groups:
    group_patoms = [np.load(i) for i in group]
    ref = create_reference_patom(group_patoms)
    
    #np.save(os.path.join(output, f'patom_{ref[0,0]:.8f}.npy'), ref)

# test_group = next((sub for sub in groups if len(sub) == 10), None)
    
# test_patoms = [np.load(i) for i in test_group]

# images = [i[2:,:3] for i in test_patoms]
# vmin = min([np.nanmin(i) for i in images])
# vmax = max([np.nanmax(i) for i in images])
# norm = plt.Normalize(vmin=vmin, vmax=vmax)

# cmap = 'gray'

# fig, axes = plt.subplots(1, 20, figsize=(80, 4), constrained_layout=True)

# im = None
# for ax, img in zip(axes, images):
#     im = ax.imshow(img, cmap=cmap, norm=norm)
#     ax.axis('off')

# fig.suptitle('Example Patoms Grouped Together', fontsize=15)

# plt.show()

# # for i in images:
# #     print(i)

# ref_patom = create_reference_patom(test_patoms)
# # print(ref_patom.shape)
# ref_patom = ref_patom[2:,:3]

# plt.figure()
# plt.imshow(img, cmap=cmap)
# plt.axis('off')
# plt.title('Ref Patom Based on Group Example', fontsize=15)

# plt.show()

## show example of original frame and then rebuild reference frame
