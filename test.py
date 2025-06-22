import numpy as np
from multiprocessing import Pool, cpu_count

# -----------------------------------------------------------------------------
# 1) Define your global holder & initializer so each worker can see `arrays`
# -----------------------------------------------------------------------------
_arrays = None

def _init_worker(arrays):
    """
    Pool initializer: store the arrays list in a global variable
    inside each worker process.
    """
    global _arrays
    _arrays = arrays

# -----------------------------------------------------------------------------
# 2) Define the work each process will do on its chunk of outer-indices
# -----------------------------------------------------------------------------
def _compare_chunk(i_chunk):
    """
    i_chunk : a list (or iterable) of outer indices i
    
    For each i in i_chunk, compare _arrays[i] to _arrays[j] for j>i,
    compute whatever score or comparison you need, and collect results.
    
    Returns a list of tuples, e.g.
      (i, j, score_ij)
    """
    results = []
    local = _arrays
    n = len(local)
    
    for i in i_chunk:
        ai = local[i]
        for j in range(i+1, n):
            aj = local[j]
            
            # --- your custom comparison here ---
            # e.g. if you have a function compare_fn that takes two numpy arrays:
            score = compare_fn(ai, aj)
            # ------------------------------------
            
            results.append((i, j, score))
    return results

# -----------------------------------------------------------------------------
# 3) The driver that splits up the work, runs the pool, and collects results
# -----------------------------------------------------------------------------
def parallel_pairwise_compare(arrays, compare_fn, num_workers=None):
    """
    arrays       : list of numpy arrays
    compare_fn   : function taking (array_i, array_j) -> some score
    num_workers  : how many processes to spawn (default = all cores)
    
    Returns a flat list of (i, j, score) for all i < j.
    """
    n = len(arrays)
    if num_workers is None:
        num_workers = cpu_count()   # 8 on your i7-2760QM
    
    # 1) Decide chunk boundaries along 0..n-1
    #    e.g. if n=100_000 and workers=8, each chunk ≈12500 indices
    chunk_size = (n + num_workers - 1) // num_workers
    chunks = [
        list(range(k, min(k + chunk_size, n)))
        for k in range(0, n, chunk_size)
    ]
    
    # 2) Spawn the pool, passing `arrays` once to each worker
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(arrays,)
    ) as pool:
        # map each chunk to one process
        all_lists = pool.map(_compare_chunk, chunks)
    
    # 3) Flatten the list of lists
    flattened = [item for sublist in all_lists for item in sublist]
    return flattened

# -----------------------------------------------------------------------------
# 4) Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose you have 100 000 1D numpy arrays, each array = [id1, id2, score]
    arrays = [np.random.rand(3) for _ in range(10000)]
    
    # Your custom comparison function:
    def compare_fn(a, b):
        # for example, absolute difference of their “score” entries:
        return abs(a[2] - b[2])
    
    # Run in parallel on 8 cores:
    results = parallel_pairwise_compare(arrays, compare_fn, num_workers=4)
    
    # `results` is a list of (i, j, score_ij) for all 0 ≤ i < j < 100 000.
    print("Computed", len(results), "pairwise scores.")
