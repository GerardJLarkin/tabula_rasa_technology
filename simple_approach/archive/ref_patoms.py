# from typing import List
# import numpy as np

# def create_reference_patom(ref: np.ndarray, new_arr: np.ndarray, cnt: int) -> np.ndarray:

#     curr_ref_second_row = ref[1:2]
#     curr_ref_patom = ref[2:,:3]
#     # repeat current ref patom to facilitate calculations for updated ref patom
#     tiled_ref_second_row = np.tile(curr_ref_second_row,(cnt,1))
#     tiled_ref_patom = np.tile(curr_ref_patom,(cnt, 1))

#     new_second_row = new_arr[1:2]
#     new_patom = new_arr[2:,:3]

#     #group_first_rows  = np.vstack([pat[0:1]  for pat in patoms])
#     group_second_rows = np.vstack([tiled_ref_second_row, new_second_row])
#     group_patoms   = np.vstack([tiled_ref_patom, new_patom])

#     num_patoms = cnt + 1
#     avg_rows = int(np.ceil(group_patoms.shape[0] / num_patoms))

#     # … the rest of your unique-value, colour-processing code goes here …
#     # Just keep working off the in-memory 'data_rows' and 'second_rows'
#     # 4 columns (0, 1, 2, 3)
#     # row 1 is id, centroid coordinates and segment
#     # row 2 is min and max x and y values for original x and y coordinates in the frame
#     # remaining rows are the normalised x and y values and the normalised colour at each coordinate
    
#     x_vals, x_val_count = np.unique(group_patoms[:,0], return_counts=True)
#     x_vals = x_vals.reshape(x_vals.shape[0],1); x_val_count = x_val_count.reshape(x_val_count.shape[0],1)
#     x_vals = np.hstack((x_vals, x_val_count))
#     x_desc_order = x_vals[:,-1].argsort()[::-1]
#     x_vals_sorted = x_vals[x_desc_order]
    
#     # if the avg num rows is smaller or equal to the shape of the x values and their counts take top rows up to avg row number
#     if avg_rows <= x_vals_sorted.shape[0]:
#         x_vals = x_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
#     # if the avg num rows is greater than the shape of the x values and their counts expand the top n rows where the sum of the counts
#     # related to each top n x value is equal to the avg num rows
#     else:
#         cumsum = np.cumsum(x_vals_sorted[:,1]).reshape(x_vals_sorted.shape[0],1)
#         x_vals_cumsum = np.hstack((x_vals_sorted, cumsum))
#         counts = x_vals_cumsum[:, 1].astype(int)
#         expanded_x_vals_array = np.repeat(x_vals_cumsum, counts, axis=0)
#         x_vals = expanded_x_vals_array[:avg_rows,0].reshape(avg_rows,1)

#     y_vals, y_val_count = np.unique(group_patoms[:,1], return_counts=True)
#     y_vals = y_vals.reshape(y_vals.shape[0],1); y_val_count = y_val_count.reshape(y_val_count.shape[0],1)
#     y_vals = np.hstack((y_vals, y_val_count))
#     y_desc_order = y_vals[:,-1].argsort()[::-1]
#     y_vals_sorted = y_vals[y_desc_order]
#     if avg_rows <= y_vals_sorted.shape[0]:
#         y_vals = y_vals_sorted[:avg_rows,0].reshape(avg_rows,1)
#     else:
#         cumsum = np.cumsum(y_vals_sorted[:,1]).reshape(y_vals_sorted.shape[0],1)
#         y_vals_cumsum = np.hstack((y_vals_sorted, cumsum))
#         counts = y_vals_cumsum[:, 1].astype(int)
#         expanded_y_vals_array = np.repeat(y_vals_cumsum, counts, axis=0)
#         y_vals = expanded_y_vals_array[:avg_rows,0].reshape(avg_rows,1)

#     x_y = np.hstack((x_vals, y_vals))
    
#     #get 'average' colour at x,y postion?????
#     # back to original vstacked group of patoms, extract pixel colours for each of the x values that made it in to the final cut
#     x_colours = []
#     for i in x_y[:,0].tolist():
#         colours = group_patoms[:,2][group_patoms[:,0] == i]
#         # get mode, mean and median
#         mode_colour, colour_count = np.unique(colours, return_counts=True)
#         mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
#         mode_colour = np.hstack((mode_colour, colour_count))
#         mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
#         mode_colour = mode_colour[mode_colour_sorted,0]
#         mode_colour = mode_colour[0]
#         mean_colour = colours.mean()
#         median_colour = np.median(colours)
#         colour = (mode_colour + mean_colour + median_colour) / 3
#         x_colours.append(colour)

#     y_colours = []
#     for i in x_y[:,1].tolist():
#         colours = group_patoms[:,2][group_patoms[:,1] == i]
#         # get mode, mean and median
#         mode_colour, colour_count = np.unique(colours, return_counts=True)
#         mode_colour, colour_count = mode_colour.reshape(mode_colour.shape[0],1), colour_count.reshape(colour_count.shape[0],1)
#         mode_colour = np.hstack((mode_colour, colour_count))
#         mode_colour_sorted = np.flip(np.argsort(mode_colour[:,1]))
#         mode_colour = mode_colour[mode_colour_sorted,0]
#         mode_colour = mode_colour[0]
#         mean_colour = colours.mean()
#         median_colour = np.median(colours)
#         colour = (mode_colour + mean_colour + median_colour) / 3
#         y_colours.append(colour)

#     x_y_colours = list(zip(x_colours, y_colours))
#     x_y_colours = np.array([sum(i)/2 for i in x_y_colours]).reshape(avg_rows,1)
    
#     fill_arr = np.empty(x_y_colours.shape, dtype=float)
#     fill_arr.fill(np.nan)
#     ref_patom_values = np.column_stack((x_y, x_y_colours, fill_arr))

#     # 5) Example of saving with consistent dtype / naming
#     ref_id = np.random.default_rng().random(dtype=np.float32)
#     ref = np.vstack((np.array([[ref_id, np.nan, np.nan, np.nan]]),
#         group_second_rows.mean(axis=0, keepdims=True),
#         ref_patom_values))
    
#     return ref

import numpy as np
from collections import Counter

def create_reference_patom(ref: np.ndarray, new_arr: np.ndarray, cnt: int) -> np.ndarray:
    # 1) Weighted‐mean second row (no tiling)
    # ------------------------------------------------
    # ref[1] and new_arr[1] are shape (4,)
    total = cnt + 1
    second_row = (ref[1] * cnt + new_arr[1]) / total

    # 2) Gather all x,y,c data (no tiling)
    # ------------------------------------------------
    old_pts = ref[2:, :3]      # shape (M_old, 3)
    new_pts = new_arr[2:, :3]  # shape (M_new, 3)

    # 3) Build combined histograms of x and y
    # ------------------------------------------------
    # Use Python Counter to accumulate counts (fast for up to a few thousand pts)
    x_counts = Counter(old_pts[:,0].astype(int).tolist())   # counts per x in old
    y_counts = Counter(old_pts[:,1].astype(int).tolist())   # counts per y in old
    # scale old counts by cnt (because old_pts represented cnt times)
    for k in list(x_counts):
        x_counts[k] *= cnt
    for k in list(y_counts):
        y_counts[k] *= cnt
    # add counts from new
    for xv in new_pts[:,0].astype(int):
        x_counts[xv] += 1
    for yv in new_pts[:,1].astype(int):
        y_counts[yv] += 1

    # 4) Determine avg_rows
    # ------------------------------------------------
    combined_size = sum(x_counts.values())  # should equal cnt*len(old_pts) + len(new_pts)
    avg_rows = int(np.ceil(combined_size / (cnt + 1)))

    # 5) Select the top‐count x positions up to avg_rows
    # ------------------------------------------------
    # Sort x keys by descending count
    x_items = sorted(x_counts.items(), key=lambda kv: kv[1], reverse=True)
    sel_x = []
    running = 0
    for x_val, c in x_items:
        take = min(c, avg_rows - running)
        sel_x.extend([x_val] * take)
        running += take
        if running >= avg_rows:
            break
    sel_x = np.array(sel_x, dtype=float).reshape(-1, 1)  # shape (avg_rows,1)

    # Same for y
    y_items = sorted(y_counts.items(), key=lambda kv: kv[1], reverse=True)
    sel_y = []
    running = 0
    for y_val, c in y_items:
        take = min(c, avg_rows - running)
        sel_y.extend([y_val] * take)
        running += take
        if running >= avg_rows:
            break
    sel_y = np.array(sel_y, dtype=float).reshape(-1, 1)  # (avg_rows,1)

    # 6) Compute the combined x,y array
    # ------------------------------------------------
    x_y = np.hstack((sel_x, sel_y))  # (avg_rows, 2)

    # 7) For each selected (x,y), compute mode/mean/median colour
    # ------------------------------------------------
    # Build an index by x and y to avoid repeated searches
    # We'll filter old_pts and new_pts into one array for querying
    all_pts = np.vstack((np.repeat(old_pts, cnt, axis=0), new_pts))
    x_colours = []
    y_colours = []
    for xv, yv in x_y:
        xv = int(xv); yv = int(yv)
        mask_x = all_pts[:,0] == xv
        cols_x = all_pts[mask_x,2]
        # mode, mean, median
        vals, cnts = np.unique(cols_x, return_counts=True)
        mode = vals[np.argmax(cnts)]
        mean = cols_x.mean()
        med  = np.median(cols_x)
        x_colours.append((mode + mean + med)/3)

        mask_y = all_pts[:,1] == yv
        cols_y = all_pts[mask_y,2]
        vals, cnts = np.unique(cols_y, return_counts=True)
        mode = vals[np.argmax(cnts)]
        mean = cols_y.mean()
        med  = np.median(cols_y)
        y_colours.append((mode + mean + med)/3)

    x_colours = np.array(x_colours).reshape(-1,1)
    y_colours = np.array(y_colours).reshape(-1,1)

    # 8) Build the final per‐pixel block and stack
    # ------------------------------------------------
    colours = (x_colours + y_colours) / 2
    fill_nan = np.full_like(colours, np.nan)
    pixel_block = np.hstack((x_y, colours, fill_nan))  # shape (avg_rows,4)

    # 9) Assemble the new reference patom
    # ------------------------------------------------
    ref_id = np.random.default_rng().random(dtype=np.float32)
    top_row = np.array([[ref_id, np.nan, np.nan, np.nan]], dtype=np.float32)
    second = second_row.astype(np.float32).reshape(1,4)
    final = np.vstack((top_row, second, pixel_block.astype(np.float32)))

    return final
