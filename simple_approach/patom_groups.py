import numpy as np
from functools import lru_cache
from time import perf_counter
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.reference import create_reference_patom

@lru_cache(maxsize=10000)
def load_memmap(path):
    return np.load(path, mmap_mode='r')

def compare_reference_thread(arr_new, reference_array, threshold, compare_func):
    _, _, score = compare_func(arr_new, reference_array)
    return score <= threshold

def process_batch(file_batch, threshold, compare_func):
    local_groups = []
    local_references = []

    for path in file_batch:
        arr_new = load_memmap(path)
        placed = False

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {
                executor.submit(compare_reference_thread, arr_new, ref_arr, threshold, compare_func): idx
                for idx, ref_arr in enumerate(local_references)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                if future.result():
                    local_groups[idx].append(path)
                    member_arrays = [load_memmap(p) for p in local_groups[idx]]
                    local_references[idx] = create_reference_patom(member_arrays)
                    placed = True
                    break

        if not placed:
            local_groups.append([path])
            local_references.append(arr_new)

    return local_groups, local_references

def group_arrays_multiprocess(file_paths, compare_func, threshold, batch_size=500):
    total_start = perf_counter()
    batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(process_batch, [(batch, threshold, compare_func) for batch in batches])

    # Flatten local groups & references
    groups = []
    references = []
    for local_groups, local_references in results:
        groups.extend(local_groups)
        references.extend(local_references)

    # Merge similar groups (coarse pass)
    merged_groups = []
    merged_references = []

    for idx, ref in enumerate(references):
        arr_new = ref
        placed = False

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_index = {
                executor.submit(compare_reference_thread, arr_new, m_ref, threshold, compare_func): i
                for i, m_ref in enumerate(merged_references)
            }

            for future in as_completed(future_to_index):
                i = future_to_index[future]
                if future.result():
                    merged_groups[i].extend(groups[idx])
                    member_arrays = [load_memmap(p) for p in merged_groups[i]]
                    merged_references[i] = create_reference_patom(member_arrays)
                    placed = True
                    break

        if not placed:
            merged_groups.append(groups[idx])
            merged_references.append(ref)

    total_elapsed = (perf_counter() - total_start) / 60
    print(f'Total clustering complete in {total_elapsed:.1f} mins; formed {len(merged_groups)} groups.')

    return merged_references
