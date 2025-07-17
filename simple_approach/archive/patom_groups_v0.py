import numpy as np
import multiprocessing as mp
import gc
from typing import List, Callable

# This will live in each worker
_worker_compare: Callable[[np.ndarray, np.ndarray], float] = None

def _init_worker(compare_func: Callable[[np.ndarray, np.ndarray], float]):
    """
    Pool initializer: stash the user's compare_func in a global
    so child processes can call it.
    """
    global _worker_compare
    _worker_compare = compare_func

def _test_group_membership(args):
    """
    Worker: given a new‐array path + one group's file‐paths + threshold,
    return True if new arr is within threshold of *all* members.
    """
    new_path, group_paths, threshold = args
    # memmap the “new” array
    arr_new = np.load(new_path, mmap_mode='r')

    for member_path in group_paths:
        arr_existing = np.load(member_path, mmap_mode='r')
        id1, id2, diff = _worker_compare(arr_new, arr_existing)
        # teardown before next iteration
        del arr_existing
        gc.collect()
        if diff > threshold:
            del arr_new
            gc.collect()
            return False   # Fails complete‐link test

    # if we get here, it was within threshold of every member
    del arr_new
    gc.collect()
    return True

def group_arrays(
    file_paths: List[str],
    compare_func: Callable[[np.ndarray, np.ndarray], float],
    threshold: float,
    num_workers: int = None
) -> List[List[str]]:
    """
    Same API as before, but tests each existing group in parallel.

    Parameters
    ----------
    file_paths
        list of .npy file paths to cluster.
    compare_func
        function(arr1, arr2) -> numeric difference.
    threshold
        max‐allowed difference.
    num_workers
        how many processes to spin up. If None, uses os.cpu_count().

    Returns
    -------
    List of clusters, each a list of file‐paths.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    num_patoms_left = len(file_paths)

    groups: List[List[str]] = []

    # Build the pool up front, so we reuse workers & their memmaps
    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(compare_func,)
    ) as pool:
        for new_path in file_paths:
            placed = False
            print(num_patoms_left)
            num_patoms_left -= 1

            if groups:
                # Prepare one task per existing group
                tasks = [
                    (new_path, grp, threshold)
                    for grp in groups
                ]
                # run in parallel
                results: List[bool] = pool.map(_test_group_membership, tasks)

                # assign to the first group that passed
                for grp, fits in zip(groups, results):
                    if fits:
                        grp.append(new_path)
                        placed = True
                        break

            if not placed:
                # start a brand-new group
                groups.append([new_path])

    return groups
