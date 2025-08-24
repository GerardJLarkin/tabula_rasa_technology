import os
import sys
import glob
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple, Set
from collections import defaultdict
import pickle

# START: CHATGPT GENERATED CODE #
# --- minimise BLAS thread fan-out (set before numpy import) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# END: CHATGPT GENERATED CODE #

import numpy as np
from multiprocessing import Pool, cpu_count

# import required functions
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
from tabula_rasa_technology.simple_approach.reference import create_reference_patom
from tabula_rasa_technology.simple_approach.compare import compare

# folder paths
root = os.path.dirname(__file__)
folder = os.path.join(root, 'test_historic_data')
output = os.path.join(root, 'test_reference_patoms')

# START: CHATGPT GENERATED CODE #
# ----------------------- configuration -----------------------
SIM_THRESHOLD: float = 0.01
BLOCK: int = 1024  # tile edge for upper-triangle tiling

# ---------------------- grouping helper ----------------------
class MultiGroupBuilder:
    
    def __init__(self, capacity: int = 0):
        self.groups: List[List[Any]] = [[] for _ in range(capacity)]
        self.group_members: List[Set[Any]] = [set() for _ in range(capacity)]
        self.item_to_groups: Dict[Any, Set[int]] = defaultdict(set)

    def _add_to_group(self, g: int, x: Any):
        if x not in self.group_members[g]:
            self.group_members[g].add(x)
            self.groups[g].append(x)
            self.item_to_groups[x].add(g)

    def add_pair(self, a: Any, b: Any):
        targets = self.item_to_groups.get(a, set()) | self.item_to_groups.get(b, set())
        for g in targets:
            self._add_to_group(g, a)
            self._add_to_group(g, b)

    def compact(self) -> List[List[Any]]:
        return [lst for lst in self.groups if lst]

# -------------------- worker-side cache ----------------------
# Avoid sending big numpy objects through pickling. Instead, pass only paths.
# Each process lazily mmaps files on first use and reuses the memmap thereafter.
_G_PATHS: List[Path] = []
_G_THRESHOLD: float = 0.0
_G_CACHE: Dict[int, np.ndarray] = {}

def _init_worker(paths: List[Path], threshold: float):
    global _G_PATHS, _G_THRESHOLD, _G_CACHE
    _G_PATHS = paths
    _G_THRESHOLD = threshold
    _G_CACHE = {}

def _get_arr(idx: int) -> np.ndarray:
    arr = _G_CACHE.get(idx)
    if arr is None:
        # mmap: minimal RSS; data paged in on demand during compare()
        arr = np.load(str(_G_PATHS[idx]), mmap_mode='r')
        _G_CACHE[idx] = arr
    return arr

# --------------------- pair tiling & work --------------------
def pairwise_tiles(n: int, block: int = BLOCK):
    """Yield (i0,i1,j0,j1) tiles covering the upper triangle (inclusive of diagonal)."""
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        for j0 in range(i0, n, block):
            j1 = min(n, j0 + block)
            yield (i0, i1, j0, j1)

def _process_tile(args: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    i0, i1, j0, j1 = args
    good: List[Tuple[int, int]] = []
    for i in range(i0, i1):
        di = _get_arr(i)
        j_start = max(j0, i + 1)  # upper triangle only (j>i)
        for j in range(j_start, j1):
            dj = _get_arr(j)
            # compare() returns (id1, id2, score) – leave as-is
            _, _, score = compare(di, dj)
            if score <= _G_THRESHOLD:
                good.append((i, j))
    return good

# -------------------------- driver ---------------------------
def build_groups_from_pairs(paths: List[Path], threshold: float = SIM_THRESHOLD) -> List[List[Path]]:
    n = len(paths)
    gb = MultiGroupBuilder(capacity=n)
    tiles = list(pairwise_tiles(n, BLOCK))
    t0 = perf_counter()
    with Pool(processes=6, initializer=_init_worker, initargs=(paths, threshold)) as pool:
        for k, pairs in enumerate(pool.imap_unordered(_process_tile, tiles, chunksize=8), start=1):
            for i, j in pairs:
                gb.add_pair(paths[i], paths[j])
            if k % 10 == 0 or k == 1:
                elapsed = perf_counter() - t0
                rate = round((elapsed)/60, 2)
                approx_pairs = sum(len(g) for g in gb.groups)
                print(f"[tiles] {k}/{len(tiles)} done ({rate:.2f} time (mins), ~{approx_pairs} items in groups)")
    return gb.compact()
# END: CHATGPT GENERATED CODE #

# read in patom file paths
paths = glob.glob(folder + '/*.npy')

# build patom groups
groups = build_groups_from_pairs(paths, SIM_THRESHOLD)
print(f"Formed {len(groups)} groups")

# save groups for testing later
with open("groups.pkl", "wb") as f:
    pickle.dump(groups, f, protocol=pickle.HIGHEST_PROTOCOL)

# create and save reference patoms
for group in groups:
    group_patoms = [np.load(str(p), mmap_mode='r') for p in group]
    ref_patom = create_reference_patom(group_patoms)
    np.save(f'test_reference_patoms/patom_{ref_patom[0,0]:.8f}.npy"', ref_patom)
