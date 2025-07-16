import numpy as np
import gc

def group_arrays(file_paths, compare_func, threshold):
    """
    Groups arrays by streaming through disk; only two arrays in memory at once.

    Parameters
    ----------
    file_paths : list of str
        Paths to .npy files on disk.
    compare_func : callable
        Your bespoke function: compare_func(arr1, arr2) -> a numeric “difference”.
    threshold : float
        Maximum allowed difference to consider two arrays “similar”.

    Returns
    -------
    List[list[str]]
        A list of groups; each group is a list of file paths.
    """
    groups = []         # list of lists of file paths
    # We won’t keep whole arrays in RAM—just file‐paths for group members.
    
    num_patoms_left = len(file_paths)

    for path in file_paths:
        # Memory-map the “new” array (no full load)
        arr_new = np.load(path, mmap_mode='r')
        placed = False
        print(num_patoms_left)
        num_patoms_left -= 1

        # Try to insert into an existing group
        for grp in groups:
            # For complete‐link clustering, compare to ALL members in grp.
            is_similar_to_all = True
            for member_path in grp:
                arr_existing = np.load(member_path, mmap_mode='r')
                id_a, id_b, score = compare_func(arr_new, arr_existing)
                # as soon as one member is too far, break
                if score > threshold:
                    is_similar_to_all = False
                    # free the memmap of existing before breaking
                    del arr_existing
                    gc.collect()
                    break
                # free this memmap before next iteration
                del arr_existing
                gc.collect()

            if is_similar_to_all:
                grp.append(path)
                placed = True
                break

        # If it doesn’t fit anywhere, start a new group
        if not placed:
            groups.append([path])

        # free the memmap of the new array
        del arr_new
        gc.collect()

    return groups
