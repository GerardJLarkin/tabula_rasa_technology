# end to end process
# cd ~/Desktop/capstone_project/tabula_rasa_technology/simple_approach
# source .venv/bin/activate
# …run your scripts, imports, etc…

import numpy as np
from multiprocessing import Pool
from time import perf_counter
from typing import Callable, List
from numba import njit
import sys
import os

import multiprocessing as mp
mp.set_start_method('fork') 

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

root = os.path.dirname(os.path.abspath(__file__))
# load reference patoms
reference = os.path.join(root, 'reference_patoms')

from tabula_rasa_technology.simple_approach.patoms_test import patoms
from tabula_rasa_technology.simple_approach.compare_test import compare
from tabula_rasa_technology.simple_approach.ref_patom_test import RefPatom
from tabula_rasa_technology.simple_approach.ref_patoms import create_reference_patom


# class GroupManager:
#     def __init__(self, dim, thresh):
#         self.refs = np.zeros((0, dim), dtype=float)
#         self.counts = np.zeros((0,), dtype=int)
#         self.refs_obj = []    # list of RefPatom instances
#         self.thresh = thresh

#     def add_batch(self, patom_arrs: list[np.ndarray]):
#         N = len(patom_arrs)
#         feats = np.stack([patom[2:,:3].ravel() for patom in patom_arrs])  # flatten to 1D
#         norms = np.linalg.norm(feats, axis=1)
#         if self.refs.shape[0]:
#             ref_feats = np.stack([r.reference()[2:,:3].ravel() for r in self.refs_obj])
#             ref_norms = np.linalg.norm(ref_feats, axis=1)
#             sims = (ref_feats @ feats.T) / (ref_norms[:,None] * norms[None,:])
#             best_idx = sims.argmax(axis=0)
#             best_sim = sims.max(axis=0)
#         else:
#             best_idx = np.full(N, -1, dtype=int)
#             best_sim = np.full(N, -np.inf, dtype=float)

#         for i, patom in enumerate(patom_arrs):
#             idx, sim = best_idx[i], best_sim[i]
#             if sim >= self.thresh:
#                 self.refs_obj[idx].update(patom)
#             else:
#                 new_ref = RefPatom(patom)
#                 self.refs_obj.append(new_ref)

#     def finalize(self):
#         # save all references
#         for i, r in enumerate(self.refs_obj):
#             np.save(f"ref_{i}.npy", r.reference())


class GroupManager:
    def __init__(self,
        compare_fn: Callable[[np.ndarray, np.ndarray], float],
        create_ref_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
        threshold: float):
        """
        compare_fn(ref_patom: np.ndarray, new_patom: np.ndarray) -> float
        ref_cls is your RefPatom (or ArrayGroup) class with update() and reference() methods
        """
        self.compare_fn = compare_fn
        self.create_ref_fn = create_ref_fn
        self.threshold  = threshold
        self.refs:  List[np.ndarray] = []  # current reference patoms
        self.counts: List[int]       = []  # how many have been merged
    
    def add_patom(self, patom: np.ndarray):
        # 1) empty → new group
        if not self.refs:
            self.refs.append(patom)
            self.counts.append(1)
            return

        # 2) find best match
        best_sim = -np.inf
        best_i   = -1
        for i, old_ref in enumerate(self.refs):
            id1, id2, sim = self.compare_fn(old_ref, patom)
            if sim < best_sim:
                best_sim, best_i = sim, i

        # 3) update or spawn
        if best_sim >= self.threshold:
            cnt = self.counts[best_i]
            new_ref = self.create_ref_fn(old_ref=self.refs[best_i],
                                         new_arr=patom,
                                         cnt=cnt)
            self.refs[best_i]   = new_ref
            self.counts[best_i] = cnt + 1
        else:
            self.refs.append(patom)
            self.counts.append(1)
    
    def add_batch(self, patoms: List[np.ndarray]):
        for p in patoms:
            self.add_patom(p)

    def get_references(self) -> List[np.ndarray]:
        return self.refs

    def get_counts(self) -> List[int]:
        return self.counts

    # def finalize(self):
    #     # e.g. save each reference
    #     for i, grp in enumerate(self.groups):
    #         ref = grp.reference()
    #         np.save(f"ref_{i}.npy", ref)


start = perf_counter()
def main():
    # 1) Build one Pool here
    with Pool(processes=4) as pool:
        # load and flatten your Moving-MNIST or frame sequence
        data = np.load('mnist_test_seq.npy')   # (10k,20,64,64)
        frames = data.reshape(-1,64,64)[:2000]/255.0

        mgr = GroupManager(compare, RefPatom, threshold=0.25)  # example dim
        for ix, frame in enumerate(frames):
            pat_list = patoms(frame, pool=pool)
            mgr.add_batch(pat_list)
            print('complete frame:', ix, 'time to complete (mins):', round((perf_counter()-start)/60,2))

        refs =  mgr.get_references()

        for i, ref_patom in enumerate(refs):
            fname = os.path.join(reference, f"ref_patom_{i:08d}.npy")
            np.save(fname, ref_patom)

if __name__ == "__main__":
    main()
