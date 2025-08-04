# end to end process
# cd ~/Desktop/capstone_project/tabula_rasa_technology/simple_approach
# source .venv/bin/activate
# …run your scripts, imports, etc…


## Part 1: function to find patterns in numpy array
import numpy as np
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from time import perf_counter, process_time, clock_gettime_ns, CLOCK_REALTIME
import random
import string

_POOL = Pool(processes=4)

threshold = 0.008 #0.00005
motion = np.array([[-1, -1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]])

def snapshot(single_frame_array, i):

    orig_array = single_frame_array
    
     # example: compute all eight neighbour differences at once
    core = orig_array[1:-1,1:-1]
    neighbors = np.stack([orig_array[1+di:1+di+core.shape[0], 1+dj:1+dj+core.shape[1]] for di,dj in motion], axis=0)     # shape (8, H-2, W-2)
    diffs = np.abs(neighbors - core[None])
    truth = diffs <= threshold  # boolean array (8, H-2, W-2)
    # now extract indices in one shot, etc.

    true_indices = np.asarray(truth).nonzero() 

    di, dj = motion[i]
    orig_loc_i = true_indices[0] + di
    orig_loc_j = true_indices[1] + dj

    orig_vals = orig_array[orig_loc_i, orig_loc_j]

    originals = np.column_stack((orig_vals, orig_loc_i, orig_loc_j))
    
    tnn_loc_i = true_indices[0]
    tnn_loc_j = true_indices[1]
    tnn_vals = orig_array[true_indices[0], true_indices[1]]
    
    nearest_neigbours = np.column_stack((tnn_vals, tnn_loc_i, tnn_loc_j))

    orig_nn = np.vstack((originals, nearest_neigbours))
    
    return orig_nn

def patoms(single_frame_array):
    items = [(single_frame_array, i) for i in range(8)]
    res = _POOL.starmap(snapshot, items)

    # combine the outputs of each nearest neighbour function
    combined_output = np.vstack((res))
    combined_output = np.unique(combined_output, axis=0)
    combined_output = combined_output[combined_output[:,0].argsort()]

    ######################################################################
    ######################################################################
    ## adding in section to save combined oupput to file for inspection ##
    ## only saves last file (all others are overwritten)                ##
    np.savetxt(
    "inspection_data_combined_output.csv",     # output file
    combined_output,            # the array to save
    delimiter=",",  # comma‐separated
    fmt="%.18e"     # format each float; adjust as needed
    )

    
    # split patoms based on colour threshold
    differences = np.diff(combined_output[:, 0])
    split_indices = np.where(differences > threshold)[0] + 1
    chunks = np.split(combined_output, split_indices)
    
    norm_patoms = []
    for i in chunks:
        center_x = (i.shape[0]-1)/2; center_y = (i.shape[1]-1)/2 # scalar will never change
        num_segments = 16 # scalar will never change
        segment_width = 360 / num_segments
        # skip over and don't save/return patoms that take up 70% or more of the array pixel number
        # only included as my laptop does not have ability to handle large volumes of data
        # if i.shape[0] >= ((single_frame_array.shape[0] * single_frame_array.shape[1]) * 0.7):
        #     pass
        # # # ignore patoms that are less than 2 pixels in size (this is 1/2000 of the input array) (skipping detail but laptop can't proccess otherwise)
        # elif i.shape[0] <= 3:
        #     pass
        # else:
        x_vals = i[:,1]; y_vals = i[:,2]
        
        x_mean = np.floor(x_vals.mean()); y_mean = np.floor(y_vals.mean())
        min_x = x_vals.min(); max_x = x_vals.max(); denominator_x = max_x - min_x
        adj_denom_x = np.where(denominator_x == 0, 1, denominator_x)
        norm_x = 2 * ((x_vals - x_vals.min()) / adj_denom_x) - 1
        
        min_y = y_vals.min(); max_y = y_vals.max(); denominator_y = max_y - min_y
        adj_denom_y = np.where(denominator_y == 0, 1, denominator_y)
        norm_y = 2 * ((y_vals - y_vals.min()) / adj_denom_y) - 1
        
        colours = i[:,0]
        
        angle_deg = (np.degrees(np.arctan2(center_y - y_mean, x_mean - center_x)) + 360) % 360
        angle_clockwise_from_north = (90 - angle_deg) % 360
        segment = angle_clockwise_from_north // segment_width
        
        patom_id = np.random.default_rng().random(dtype=np.float32)

        ## column stack for id, cent_x, cent_y, segment
        first_row = np.array([patom_id, x_mean, y_mean, segment])
        ## column stack for min x, max x, min y, max y
        second_row = np.array([min_x, max_x, min_y, max_y])
        ## column stack for norm x, norm y, colours, ...
        patom_values = np.column_stack((norm_x, norm_y, colours))
        patom_padded = np.full((patom_values.shape[0], 1), np.nan)
        patom_values = np.hstack([patom_values, patom_padded])

        # 4 columns (0, 1, 2, 3)
        # row 1 is id, centroid coordinates and segment
        # row 2 is min and max x and y values for original x and y coordinates in the frame
        # remaining rows are the normalised x and y values and the normalised colour at each coordinate
        patom_array = np.vstack((first_row, second_row, patom_values)).astype('float32')
        norm_patoms.append(patom_array)
    
    return norm_patoms

## Part 2: function to compare numpy arrays
import numpy as np
from typing import Any, List, Union

def compare(a: np.ndarray, b: np.ndarray) -> List[Union[int, float, Any]]:
    # 4 columns (0, 1, 2, 3)
    # row 1 is id, centroid coordinates and segment
    # row 2 is min and max x and y values for original x and y coordinates in the frame
    # remaining rows are the normalised x and y values and the normalised colour at each coordinate
    
    # IDs
    id_a, id_b = a[0, 0], b[0, 0]

    # get normalise coordinates
    pos1 = a[2:, :2].astype(np.float32, copy=False)
    pos2 = b[2:, :2].astype(np.float32, copy=False)

    m, n = pos1.shape[0], pos2.shape[0]

    # Pixel-count similarity
    fill_diff = abs(m - n) / ((m + n) / 2)
    fill_sim = min(fill_diff, 1.0)
    
    # adding in this heuristic check to attempt to improve processing efficiency - don't want to do it as the same shape (big vs small circle) can
    # have very different number pixels
    # (this does not significantly reduce the processing overhead when generating the reference patoms, therefore not being used)
    # if fill_sim >= 0.5:
    #     score = 1.0
    # else:
    
    dists = np.linalg.norm(pos1[:,None,:] - pos2[None,:,:], axis=2)
    dists_denom = dists.max() - dists.min()
    adj_dists_denom = np.where(dists_denom == 0, 1, dists_denom)
    dists_norm = (dists - dists.min()) / adj_dists_denom
    pos_sim = dists_norm.mean()
    
    # get normalised colour values
    col1 = a[2:, 2].astype(np.float32, copy=False)
    col2 = b[2:, 2].astype(np.float32, copy=False)
    # Colour similarity: sum |c_i - d_j| over all i,j
    colour_sim = np.abs(col1[:, None] - col2[None, :]).sum() / (m * n)

    # weighted combination
    # total = pos_sim + colour_sim + fill_sim
    total = 7
    if (pos_sim + colour_sim + fill_sim) == 0.0:
        score = 0.0
        #print('exact', 0.0)
    elif (pos_sim <= 0.2) and (fill_sim <= 0.2):
        score = 0.2
        #print('pos fill', 0.2)
    else:
        score = ((pos_sim * 4) + (colour_sim * 1) + (fill_sim * 2)) / total
        #print('other', score)

    return [id_a, id_b, score]

## Part 3: function to create reference patterns
from typing import List
import numpy as np

def create_reference_patom(ref: np.ndarray, new_arr: np.ndarray, cnt: int) -> np.ndarray:

    curr_ref_second_row = ref[1:2]
    curr_ref_patom = ref[2:,:3]
    # repeat current ref patom to facilitate calculations for updated ref patom
    tiled_ref_second_row = np.tile(curr_ref_second_row,(cnt,1))
    tiled_ref_patom = np.tile(curr_ref_patom,(cnt, 1))

    new_second_row = new_arr[1:2]
    new_patom = new_arr[2:,:3]

    #group_first_rows  = np.vstack([pat[0:1]  for pat in patoms])
    group_second_rows = np.vstack([tiled_ref_second_row, new_second_row])
    group_patoms   = np.vstack([tiled_ref_patom, new_patom])

    num_patoms = cnt + 1
    avg_rows = int(np.ceil(group_patoms.shape[0] / num_patoms))

    # … the rest of your unique-value, colour-processing code goes here …
    # Just keep working off the in-memory 'data_rows' and 'second_rows'
    # 4 columns (0, 1, 2, 3)
    # row 1 is id, centroid coordinates and segment
    # row 2 is min and max x and y values for original x and y coordinates in the frame
    # remaining rows are the normalised x and y values and the normalised colour at each coordinate
    
    x_vals, x_val_count = np.unique(group_patoms[:,0], return_counts=True)
    x_vals = x_vals.reshape(x_vals.shape[0],1); x_val_count = x_val_count.reshape(x_val_count.shape[0],1)
    x_vals = np.hstack((x_vals, x_val_count))
    x_desc_order = x_vals[:,-1].argsort()[::-1]
    x_vals_sorted = x_vals[x_desc_order]
    
    # if the avg num rows is smaller or equal to the shape of the x values and their counts take top rows up to avg row number
    if avg_rows <= x_vals_sorted.shape[0]:
        x_vals = x_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
    # if the avg num rows is greater than the shape of the x values and their counts expand the top n rows where the sum of the counts
    # related to each top n x value is equal to the avg num rows
    else:
        cumsum = np.cumsum(x_vals_sorted[:,1]).reshape(x_vals_sorted.shape[0],1)
        x_vals_cumsum = np.hstack((x_vals_sorted, cumsum))
        counts = x_vals_cumsum[:, 1].astype(int)
        expanded_x_vals_array = np.repeat(x_vals_cumsum, counts, axis=0)
        x_vals = expanded_x_vals_array[:avg_rows,0].reshape(avg_rows,1)

    y_vals, y_val_count = np.unique(group_patoms[:,1], return_counts=True)
    y_vals = y_vals.reshape(y_vals.shape[0],1); y_val_count = y_val_count.reshape(y_val_count.shape[0],1)
    y_vals = np.hstack((y_vals, y_val_count))
    y_desc_order = y_vals[:,-1].argsort()[::-1]
    y_vals_sorted = y_vals[y_desc_order]
    if avg_rows <= y_vals_sorted.shape[0]:
        y_vals = y_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
    else:
        cumsum = np.cumsum(y_vals_sorted[:,1]).reshape(y_vals_sorted.shape[0],1)
        y_vals_cumsum = np.hstack((y_vals_sorted, cumsum))
        counts = y_vals_cumsum[:, 1].astype(int)
        expanded_y_vals_array = np.repeat(y_vals_cumsum, counts, axis=0)
        y_vals = expanded_y_vals_array[:avg_rows,0].reshape(avg_rows,1)

    x_y = np.hstack((x_vals, y_vals))
    
    #get 'average' colour at x,y postion?????
    # back to original vstacked group of patoms, extract pixel colours for each of the x values that made it in to the final cut
    x_colours = []
    for i in x_y[:,0].tolist():
        colours = group_patoms[:,2][group_patoms[:,0] == i]
        # get mode, mean and median
        mode_colour, colour_count = np.unique(colours, return_counts=True)
        mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
        mode_colour = np.hstack((mode_colour, colour_count))
        mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
        mode_colour = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0]
        mean_colour = colours.mean()
        median_colour = np.median(colours)
        colour = (mode_colour + mean_colour + median_colour) / 3
        x_colours.append(colour)

    y_colours = []
    for i in x_y[:,1].tolist():
        colours = group_patoms[:,2][group_patoms[:,1] == i]
        # get mode, mean and median
        mode_colour, colour_count = np.unique(colours, return_counts=True)
        mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
        mode_colour = np.hstack((mode_colour, colour_count))
        mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
        mode_colour = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0]
        mean_colour = colours.mean()
        median_colour = np.median(colours)
        colour = (mode_colour + mean_colour + median_colour) / 3
        y_colours.append(colour)

    x_y_colours = list(zip(x_colours, y_colours))
    x_y_colours = np.array([sum(i)/2 for i in x_y_colours]).reshape(avg_rows,1)
    
    fill_arr = np.empty(x_y_colours.shape, dtype=float)
    fill_arr.fill(np.nan)
    ref_patom_values = np.column_stack((x_y, x_y_colours, fill_arr))

    # 5) Example of saving with consistent dtype / naming
    ref_id = np.random.default_rng().random(dtype=np.float32)
    ref = np.vstack((np.array([[ref_id, np.nan, np.nan, np.nan]]),
        group_second_rows.mean(axis=0, keepdims=True),
        ref_patom_values))
    
    return ref

## Part 4: Class/Functions to manage the creation and update of the reference patterns
from typing import Callable
import numpy as np

class ArrayGroup:
    """
    Holds one reference array + a count of how many arrays contributed to it.
    On each add, updates reference via inc_ref_fn(current_ref, new_arr, count).
    """
    def __init__(
        self,
        initial_array: np.ndarray,
        inc_ref_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray]
    ):
        self.reference = np.array(initial_array, copy=True)
        self.inc_ref_fn = inc_ref_fn
        self.count = 1

    def add_member(self, arr: np.ndarray):
        arr = np.array(arr, copy=False)
        # compute new reference from the old reference, new array, and how many we've seen
        self.reference = self.inc_ref_fn(self.reference, arr, self.count)
        self.count += 1

    def as_list(self):
        """Return [reference_array, count]"""
        return [self.reference, self.count]


class ArrayGroupManager:
    """
    Maintains multiple ArrayGroup instances.
    On add_array:
      - compares the new array to each group's reference
      - if best similarity ≥ threshold: adds to that group (and updates its reference & count)
      - otherwise makes a new group with count=1 and reference=new array
    """
    def __init__(
        self,
        compare_fn: Callable[[np.ndarray, np.ndarray], float],
        inc_ref_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
        threshold: float
    ):
        self.compare_fn = compare_fn
        self.inc_ref_fn = inc_ref_fn
        self.threshold = threshold
        self.groups: list[ArrayGroup] = []

    def add_array(self, arr: np.ndarray):
        arr = np.array(arr, copy=False)
        best_group = None
        best_sim = -np.inf

        # find the most-similar existing reference
        for group in self.groups:
            id1, id2, sim = self.compare_fn(group.reference, arr)
            if sim > best_sim:
                best_sim, best_group = sim, group

        # decide whether to join or start new
        if best_group is None or best_sim < self.threshold:
            # new reference group
            new_group = ArrayGroup(arr, self.inc_ref_fn)
            self.groups.append(new_group)
        else:
            best_group.add_member(arr)

    def get_all_groups(self):
        """
        Returns a list of [reference_array, count] for each group.
        """
        return [g.as_list() for g in self.groups]


## Part 5: combining previous functions to generate reference patterns
## import packages/libraries
from time import perf_counter
import numpy as np
import sys

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.patoms import patoms
from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.ref_patoms import create_reference_patom
from tabula_rasa_technology.simple_approach.ref_patom_mgmt import ArrayGroup, ArrayGroupManager

# -- instantiate the manager --
mgr = ArrayGroupManager(
    compare_fn=compare,
    inc_ref_fn=create_reference_patom,
    threshold=0.25
)

# number of sequences to import
n = 5
# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:n, ...]
print('loaded')
start = perf_counter()
#generate patoms from sequences
for i in range(0,n,1):
    print('seq num:', i, 'time start:', round((perf_counter()-start)/60,2))
    sequence = dataset[i]
    for j in range(0,20,1):
        frame = sequence[j] / 255.00
        
        ####### Part 2. #######
        #### create patoms ####
        #######################
        out_patoms = patoms(frame)

        ########## Part 3. ########
        #### create ref patoms ####
        ###########################
        for new_arr in out_patoms:
            mgr.add_array(new_arr)
    
    print('seq num:', i, 'time end:', round((perf_counter()-start)/60,2))  

# -- get reference patom and write to disk --
all_groups = mgr.get_all_groups()
for idx, grp in enumerate(all_groups):
    ref = grp[0]
    np.save(f'reference_patoms/patom_{str(ref[0,0])}', ref)

_POOL.close()