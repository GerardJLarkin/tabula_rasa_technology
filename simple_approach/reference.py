# function to create actual reference patom from group members
import numpy as np

def create_reference_patom(group_members):

    # 4) Split rows & stack in one go
    #group_first_rows  = np.vstack([pat[0:1]  for pat in patoms])
    group_second_rows = np.vstack([pat[1:2]  for pat in group_members])
    group_patoms   = np.vstack([pat[2: , :3] for pat in group_members])

    num_patoms = len(group_members)
    avg_rows = int(np.ceil(group_patoms.shape[0] / num_patoms))

    # … the rest of your unique-value, colour-processing code goes here …
    # Just keep working off the in-memory 'data_rows' and 'second_rows'
    # 4 columns (0, 1, 2, 3)
    # row 1 is id, centroid coordinates and segment
    # row 2 is min and max x and y values for original x and y coordinates in the frame
    # remaining rows are the normalised x and y values and the normalised colour at each coordinate
    
    x_vals, x_val_count = np.unique(group_patoms[:,0], return_counts=True)
    x_vals = np.column_stack((x_vals, x_val_count))
    x_desc_order = x_vals[:,-1].argsort()[::-1]
    x_vals_sorted = x_vals[x_desc_order]
    
    # if the avg num rows is smaller or equal to the shape of the x values and their counts take top rows up to avg row number
    if avg_rows <= x_vals_sorted.shape[0]:
        x_vals = x_vals_sorted[:avg_rows,0].reshape(-1,1)
    # if the avg num rows is greater than the shape of the x values and their counts expand the top n rows where the sum of the counts
    # related to each top n x value is equal to the avg num rows
    else:
        cumsum = np.cumsum(x_vals_sorted[:,1]).reshape(-1,1)
        x_vals_cumsum = np.hstack((x_vals_sorted, cumsum))
        counts = x_vals_cumsum[:, 1].astype(int)
        expanded_x_vals_array = np.repeat(x_vals_cumsum, counts, axis=0)
        x_vals = expanded_x_vals_array[:avg_rows,0].reshape(-1,1)

    y_vals, y_val_count = np.unique(group_patoms[:,1], return_counts=True)
    y_vals = np.column_stack((y_vals, y_val_count))
    y_desc_order = y_vals[:,-1].argsort()[::-1]
    y_vals_sorted = y_vals[y_desc_order]
    if avg_rows <= y_vals_sorted.shape[0]:
        y_vals = y_vals_sorted[:avg_rows,0].reshape(-1,1)
    else:
        cumsum = np.cumsum(y_vals_sorted[:,1]).reshape(-1,1)
        y_vals_cumsum = np.hstack((y_vals_sorted, cumsum))
        counts = y_vals_cumsum[:, 1].astype(int)
        expanded_y_vals_array = np.repeat(y_vals_cumsum, counts, axis=0)
        y_vals = expanded_y_vals_array[:avg_rows,0].reshape(-1,1)

    x_y = np.hstack((x_vals, y_vals))
    
    def compute_colour_stat(values, coords, targets):
        results = np.empty((targets.shape[0],), dtype=np.float32)
        for idx, val in enumerate(targets[:, 0]):
            mask = coords == val
            colours = values[mask]
            if colours.size == 1:
                results[idx] = colours[0]
                continue
            unique_colours, counts = np.unique(colours, return_counts=True)
            mode = unique_colours[np.argmax(counts)]
            mean = colours.mean()
            median = np.median(colours)
            results[idx] = (mode + mean + median) / 3
        return results.reshape(-1, 1)

    x_colours = compute_colour_stat(group_patoms[:, 2], group_patoms[:, 0], x_vals)
    y_colours = compute_colour_stat(group_patoms[:, 2], group_patoms[:, 1], y_vals)

    x_y = np.hstack((x_vals, y_vals))
    x_y_colours = (x_colours + y_colours) / 2
    ref_patom_values = np.hstack((x_y, x_y_colours, np.full_like(x_y_colours, np.nan)))
    
    ref_id = np.random.default_rng().random(dtype=np.float32)
    reference_patom = np.vstack((
        np.array([[ref_id, np.nan, np.nan, np.nan]]),
        group_second_rows.mean(axis=0, keepdims=True), 
        ref_patom_values))

    return reference_patom