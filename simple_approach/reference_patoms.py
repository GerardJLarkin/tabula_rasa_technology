# create reference patoms
import numpy as np
import os
import glob
import sys
import random
import string
from itertools import islice
from multiprocessing import Pool
from time import perf_counter

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(root, 'historic_data')

from tabula_rasa_technology.simple_approach.compare_v1 import compare
from tabula_rasa_technology.simple_approach.patom_groups import group_arrays

# Getting all the numpy arrays .npy files based on matching pattern (*.npy)
file_paths = glob.glob(os.path.join(folder, '*.npy'))
file_paths = file_paths[:1000]
# set a similarity threshold
sim_threshold = 0.25

groups = group_arrays(file_paths, compare, sim_threshold)

print('groups', len(groups))

for group in groups:
    patoms = [np.load(os.path.join(folder, fname), allow_pickle=True) for fname in group]
    # need to split each patom into 3 parts: 1st row, 2nd row, all other rows
    group_first_rows = [pat[1,:] for pat in patoms]
    group_second_rows = [pat[2,:] for pat in patoms]
    group_patoms = [pat[2:,:3] for pat in patoms]
    num_patoms = len(group_patoms)
    
    # stack first rows to create a singluar group patom numpy array
    group_first_rows = tuple(group_first_rows)
    group_first_rows = np.vstack(group_first_rows)

    # stack second rows to create a singluar group patom numpy array
    group_second_rows = tuple(group_second_rows)
    group_second_rows = np.vstack(group_second_rows)

    # stack patoms to create a singluar group patom numpy array
    group_patoms = tuple(group_patoms)
    group_patoms = np.vstack(group_patoms)
    
    # i'm ok with average number of rows for now
    avg_rows = int(np.ceil(group_patoms.shape[0] / num_patoms))
    
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

    # get avg min, max x - get second row of each patom?
    group_min_x = group_second_rows[:,0].mean()
    group_max_x = group_second_rows[:,1].mean()

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
    
    # get avg min, max y
    group_min_y = group_second_rows[:,2].mean()
    group_max_y = group_second_rows[:,3].mean()

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
        mode_color = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0,0]
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
        mode_color = mode_colour[mode_colour_sorted,0]
        mode_colour = mode_colour[0,0]
        mean_colour = colours.mean()
        median_colour = np.median(colours)
        colour = (mode_colour + mean_colour + median_colour) / 3
        y_colours.append(colour)

    x_y_colours = list(zip(x_colours, y_colours))
    x_y_colours = np.array([sum(i)/2 for i in x_y_colours]).reshape(avg_rows,1)

    # create a reference patom id
    ref_patom_id = np.random.default_rng().random(dtype=np.float32)

    first_row = np.array([ref_patom_id, np.nan, np.nan, np.nan]).reshape(1,4)
    second_row = np.array([group_min_x, group_max_x, group_min_y, group_max_y]).reshape(1,4)
    ref_patom = np.hstack((x_y, x_y_colours))
    ref_patom_padded = np.full((ref_patom.shape[0], 1), np.nan)
    ref_patom_values = np.hstack([ref_patom, ref_patom_padded])

    ref_patom = np.vstack((first_row, second_row, ref_patom_values))
    
    np.save(f'reference_patoms/patom_{str(ref_patom_id)}', ref_patom)

end = perf_counter()
print("Time taken (mins):", (end - start)/60)