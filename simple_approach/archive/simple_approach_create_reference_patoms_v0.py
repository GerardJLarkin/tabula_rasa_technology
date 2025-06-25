# create reference patoms
import numpy as np
import os
import glob
import sys
import math
from scipy import stats
import random
import string
from itertools import combinations, islice
from multiprocessing import Pool, cpu_count
import networkx as nx
from time import perf_counter

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(root, 'historic_data')

from simple_approach_compare_v0 import compare

# Getting all the numpy arrays .npy files based on matching pattern (*.npy)
file_paths = glob.glob(os.path.join(folder, '*.npy'))
# Import arrays from folder and store them as a dict
patoms = [np.load(f, allow_pickle=True) for f in file_paths][:30]
print('loaded')

# set a similarity threshold
sim_threshold = 0.3
total = len(patoms)
patom_ids = [i[0,0] for i in patoms]
s = perf_counter()
# create an empty list to hold the patim ids and their similarity score
similar_patoms = []
# see if multiprocessing has an impact - maybe not as arrays are quite small
with Pool() as pool:
    for a, b in combinations(patoms, 2):
        items = [(a, b)]
        for k, results in enumerate(pool.starmap(compare, items), start=1):
            if k % 10 == 0 or k == total:
                elapsed = (perf_counter() - s)
                print(f"{k}/{total} done. Time taken {elapsed:.1f}s", flush=True)
            if results[2] <= sim_threshold:
                similar_patoms.append(results[:2])

# delete patoms list from memory
del patoms

# check which patoms have not been added to the similar patoms lists
not_similar_patoms = []
for i in patom_ids:
    if not any(i in sublist for sublist in similar_patoms):
        not_similar_patoms.append(i)


# create a reference patom for each of the patom ids in the not similar list
for i in not_similar_patoms:
    # ref patoms structure: ref_patom_id, x_vals, y_vals, colour
    ref_patom_cols = [0, 3, 4, 5]
    # red in patom from disk
    patom = np.load(os.path.join(folder, f'patom_{i}.npy'), allow_pickle=True)
    # select only required columns
    ref_patom = patom[:, ref_patom_cols]

    np.save(f'reference_patoms/patom_{i}', ref_patom)

# instantiate a graph to combine similar patoms into groups
G = nx.Graph()
for id1, id2 in similar_patoms:
    # add edges to the graph
    G.add_edge(id1, id2)

# put connected components into a set
groups = list(nx.connected_components(G))

# convert group set type to list type
groups = [list(comp) for comp in groups]
print(groups)

for idx, group in enumerate(groups):
    print(idx)
    # read in patoms from disk
    group_patoms = []
    for id in group:
        patom = np.load(os.path.join(folder, f'patom_{id}.npy'), allow_pickle=True)
        group_patoms.append(patom)
    num_patoms = len(group_patoms)

    # stack patoms to create a singluar group patom numpy array
    group_patoms = tuple(group_patoms)
    group_patoms = np.vstack(group_patoms)
    print('tot_num_rows', group_patoms.shape[0])
    print('num patoms', num_patoms)
    # i'm ok with average number of rows for now
    avg_rows = int(np.ceil(group_patoms.shape[0] / num_patoms))    
    print('avg_rows',avg_rows)
    # does the following actually make sense?????
    x_vals, x_val_count = np.unique(group_patoms[:,3], return_counts=True)
    print('x_vals 1',x_vals, 'x_val_count', x_val_count)
    x_vals = x_vals.reshape(x_vals.shape[0],1); x_val_count = x_val_count.reshape(x_val_count.shape[0],1)
    print('x_vals 2', x_vals)
    x_vals = np.hstack((x_vals, x_val_count))
    print('x_vals 3', x_vals)
    x_desc_order = x_vals[:,-1].argsort()[::-1]
    print('x_desc_order', x_desc_order)
    x_vals_sorted = x_vals[x_desc_order]
    print('x_vals_sorted', x_vals_sorted)
    
    # if the avg num rows is smaller or equal to the shape of the x values and their counts take top rows up to avg row number
    if avg_rows <= x_vals_sorted.shape[0]:
        x_vals = x_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
        print('x_vals 4', x_vals)
    # if the avg num rows is greater than the shape of the x values and their counts expand the top n rows where the sum of the counts
    # related to each top n x value is equal to the avg num rows
    else:
        cumsum = np.cumsum(x_vals_sorted[:,1]).reshape(x_vals_sorted.shape[0],1)
        print('cumsum', cumsum)
        x_vals_cumsum = np.hstack((x_vals_sorted, cumsum))
        print('x_vals_cumsum', x_vals_cumsum)
        # expand here
        counts = x_vals_cumsum[:, 1].astype(int)
        print('counts', counts)
        expanded_x_vals_array = np.repeat(x_vals_cumsum, counts, axis=0)
        print('expanded',expanded_x_vals_array)
        x_vals = expanded_x_vals_array[:avg_rows,0].reshape(avg_rows,1)
        print('x_vals 5', x_vals)
        mask_idxs = cumsum <= avg_rows
        print('mask_idxs',mask_idxs)
        max_idx = np.nonzero(mask_idxs)[0].max()
        end = min(max_idx + 2, len(x_vals_cumsum)) 
        x_vals_included = x_vals_cumsum[:end]
        counts = x_vals_included[:, 1].astype(int)
        expanded_x_vals_array = np.repeat(x_vals_included, counts, axis=0)
        

    y_vals, y_val_count = np.unique(group_patoms[:,4], return_counts=True)
    y_vals = y_vals.reshape(y_vals.shape[0],1); y_val_count = y_val_count.reshape(y_val_count.shape[0],1)
    y_vals = np.hstack((y_vals, y_val_count))
    y_desc_order = y_vals[:,-1].argsort()[::-1]
    y_vals_sorted = y_vals[y_desc_order]
    if avg_rows <= x_vals_sorted.shape[0]:
        y_vals = y_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
    else:
        cumsum = np.cumsum(y_vals_sorted[:,1]).reshape(y_vals_sorted.shape[0],1)
        y_vals_cumsum = np.hstack((y_vals_sorted, cumsum))
        mask_idxs = cumsum <= avg_rows
        max_idx = np.nonzero(mask_idxs)[0].max()
        end = min(max_idx + 2, len(y_vals_cumsum)) 
        y_vals_included = y_vals_cumsum[:end]
        counts = y_vals_included[:, 1].astype(int)
        expanded_y_vals_array = np.repeat(y_vals_included, counts, axis=0)
        y_vals = expanded_y_vals_array[:avg_rows,0].reshape(avg_rows,1)

    x_y = np.hstack((x_vals, y_vals))
    
    #get 'average' colour at x,y postion?????
    # back to original vstacked group of patoms, extract pixel colours for each of the x values that made it in to the final cut
    x_colours = []
    for i in x_y[:,0].tolist():
        #print('x',i)
        colours = group_patoms[:,5][group_patoms[:,3] == i]
        # get mode, mean and median
        mode_colour, colour_count = np.unique(colours, return_counts=True)
        mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
        mode_colour = np.hstack((mode_colour, colour_count))
        mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
        mode_color = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0,0]
        mean_colour = colours.mean()
        median_colour = np.median(colours)
        colour = (mode_colour + mean_colour + median_colour) / 3
        x_colours.append(colour)

    y_colours = []
    for i in x_y[:,1].tolist():
        #print('y', i)
        colours = group_patoms[:,5][group_patoms[:,4] == i]
        # get mode, mean and median
        mode_colour, colour_count = np.unique(colours, return_counts=True)
        mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
        mode_colour = np.hstack((mode_colour, colour_count))
        mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
        mode_color = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0,0]
        mean_colour = colours.mean()
        median_colour = np.median(colours)
        colour = (mode_colour + mean_colour + median_colour) / 3
        y_colours.append(colour)

    x_y_colours = list(zip(x_colours, y_colours))
    x_y_colours = np.array([sum(i)/2 for i in x_y_colours]).reshape(avg_rows,1)

    # create a reference patom id
    ref_patom_id = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(6))
    ref_patom_id = np.array([ref_patom_id] * avg_rows).reshape(avg_rows,1).astype('object')
    ref_patom = np.hstack((ref_patom_id, x_y, x_y_colours))
    id = ref_patom_id[0,0]
    
    np.save(f'reference_patoms/patom_{id}', ref_patom)

end = perf_counter()
print("Time taken (mins):", (end - start)/60)
