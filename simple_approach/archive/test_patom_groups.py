import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from multiprocessing import Pool, cpu_count
from time import perf_counter
import glob, os
from typing import List, Tuple
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

# your imports
from tabula_rasa_technology.simple_approach.reference import create_reference_patom
from tabula_rasa_technology.simple_approach.compare import compare

historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(__file__)
historic = os.path.join(root, 'test_historic_data')
output = os.path.join(root, 'test_reference_patoms')
os.makedirs(output, exist_ok=True)

# patom_file_paths = glob.glob(os.path.join(historic, '*.npy'))
# patom_file_paths = patom_file_paths[:8]

# ---------- your existing MultiGroupBuilder (no merges) ----------
from collections import defaultdict
from typing import Any, Callable, Set, Dict, List

class MultiGroupBuilder:
    def __init__(self, capacity: int = 0):
        self.groups: List[List[Any]] = [[] for _ in range(capacity)]
        self.group_members: List[Set[Any]] = [set() for _ in range(capacity)]
        self.item_to_groups: Dict[Any, Set[int]] = defaultdict(set)
        self.next_free = 0

    def _new_group(self) -> int:
        if self.next_free < len(self.groups):
            g = self.next_free; self.next_free += 1; return g
        self.groups.append([]); self.group_members.append(set())
        g = len(self.groups) - 1; self.next_free = g + 1; return g

    def _add_to_group(self, g: int, x: Any):
        if x not in self.group_members[g]:
            self.group_members[g].add(x)
            self.groups[g].append(x)
            self.item_to_groups[x].add(g)

    def add_pair(self, a: Any, b: Any):
        targets = self.item_to_groups.get(a, set()) | self.item_to_groups.get(b, set())
        if not targets:
            g = self._new_group()
            self._add_to_group(g, a); self._add_to_group(g, b)
        else:
            for g in targets:
                self._add_to_group(g, a); self._add_to_group(g, b)

    def compact(self) -> List[List[Any]]:
        return [lst for lst in self.groups if lst]

# ---------- config ----------
SIM_THRESHOLD = 0.05
BLOCK = 1024  # tile edge (tune 512..4096 depending on RAM/CPU)

# ---------- step 1: load files & precompute descriptors once ----------
def load_paths(folder: str, limit: int | None = None) -> List[str]:
    paths = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if limit:
        paths = paths[:limit]
    return paths

def read_in_arrays(paths: List[str]) -> List[np.ndarray]:
    patoms = []
    for p in paths:
        arr = np.load(p, mmap_mode="r")
        patoms.append(arr)
    return patoms

# ---------- step 2: multiprocessing over tiles ----------
# We'll use globals in workers to avoid pickling large objects per task
_G_DESCS: List[np.ndarray] = []
_G_THRESHOLD: float = 0.0

def _init_worker(descs, threshold):
    global _G_DESCS, _G_THRESHOLD
    _G_DESCS = descs
    _G_THRESHOLD = threshold

def _process_tile(args: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """
    Process pairs (i,j) with i in [i0,i1), j in [j0,j1), j>i.
    Return indices of pairs whose score <= threshold.
    """
    i0, i1, j0, j1 = args
    good = []
    for i in range(i0, i1):
        di = _G_DESCS[i]
        j_start = max(j0, i + 1)  # ensure upper triangle
        for j in range(j_start, j1):
            dj = _G_DESCS[j]
            # compare() returns (id1, id2, score)
            _, _, score = compare(di, dj)
            if score <= _G_THRESHOLD:
                good.append((i, j))
    return good

def pairwise_tiles(n: int, block: int = BLOCK):
    """Yield (i0,i1,j0,j1) tiles covering the upper triangle."""
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        for j0 in range(i0, n, block):
            j1 = min(n, j0 + block)
            yield (i0, i1, j0, j1)

# ---------- step 3: orchestrate ----------
def build_groups_from_pairs(paths: List[str], patoms: List[np.ndarray], threshold: float = SIM_THRESHOLD) -> List[List[str]]:
    n = len(paths)
    gb = MultiGroupBuilder(capacity=n)  # plenty of empties

    tiles = list(pairwise_tiles(n, BLOCK))
    t0 = perf_counter()
    with Pool(processes=cpu_count(), initializer=_init_worker, initargs=(patoms, threshold)) as pool:
        for k, pairs in enumerate(pool.imap_unordered(_process_tile, tiles, chunksize=1), start=1):
            # Update groups in the main process
            for i, j in pairs:
                gb.add_pair(paths[i], paths[j])
            if k % 10 == 0 or k == 1:
                done = k
                total = len(tiles)
                rate = done / max(perf_counter() - t0, 1e-9)
                print(f"[tiles] {done}/{total} done ({rate:.1f} tiles/s, pairs kept so far: ~{sum(len(g) for g in gb.groups)})", flush=True)
    return gb.compact()

import statistics
start = perf_counter()
# ---------- run ----------
if __name__ == "__main__":
    folder = historic
    paths = load_paths(folder)              # ~24k paths
    print(f"Loaded {len(paths)} paths")
    patoms = read_in_arrays(paths)      # single pass
    print("Patoms loaded into list")
    groups = build_groups_from_pairs(paths, patoms, SIM_THRESHOLD)
    print(f"Formed {len(groups)} groups")
    min_ = np.array([len(i) for i in groups]).argmin()
    min_group = groups[min_]
    #print(min_group)
    num_patoms = len(min_group)
    print(num_patoms)
    min_group_patoms = [np.load(i) for i in min_group]
    images = [i[2:,:3] for i in min_group_patoms]
    vmin = min([np.nanmin(i) for i in images])
    vmax = max([np.nanmax(i) for i in images])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cmap = 'gray' if images[0].ndim == 2 else None

    fig, axes = plt.subplots(2, num_patoms, figsize=(20, 4), constrained_layout=True)

    # im = None
    # for ax, img in zip(axes, images):
    #     im = ax.imshow(img, cmap=cmap, norm=norm)
    #     # ax.set_title(title)
    #     ax.axis('off')

    # fig.suptitle('Patoms Grouped Together', fontsize=15)

    # plt.show()

    # for i in images:
    #     print(i,'\n')

    for group in groups:
        group_patoms = [np.load(p) for p in group]
        ref_patom = create_reference_patom(group_patoms)
        np.save(os.path.join(output, f'patom_{ref_patom[0,0]:.8f}.npy'), ref_patom)

end = perf_counter()
print("time to complete (mins):", round((end-start)/60,4))