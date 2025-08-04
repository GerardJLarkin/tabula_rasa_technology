import numpy as np
from multiprocessing import Pool
from time import perf_counter
from typing import Callable
from numba import njit
import sys

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

## Part 1: function to find patterns in numpy array
def snapshot(frame: np.ndarray, threshold: float=0.008):

    """Compute all 8 neighbor‐comparisons in one go, return combined nn array."""
    # core pixels
    core = frame[1:-1, 1:-1]                                  # (H-2, W-2)
    # stack all 8 neighbor patches
    motion = [(-1,-1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0)]
    neighs = np.stack([
        frame[1+di:1+di+core.shape[0], 1+dj:1+dj+core.shape[1]]
        for di, dj in motion
    ], axis=0)                                                # (8, H-2, W-2)
    # boolean mask of similar pixels
    mask = np.abs(neighs - core[None]) <= threshold          # (8, H-2, W-2)
    # extract coords and values in one shot
    dirs, ii, jj = np.nonzero(mask)
    di = np.array([motion[d][0] for d in dirs])
    dj = np.array([motion[d][1] for d in dirs])
    orig_i, orig_j = ii + di, jj + dj
    vals_orig = frame[orig_i, orig_j]
    vals_nn   = frame[ii, jj]
    data = np.column_stack((np.concatenate([vals_orig, vals_nn]),
                             np.concatenate([orig_i, ii]),
                             np.concatenate([orig_j, jj])))
    # unique+sort once
    data = np.unique(data, axis=0)
    data = data[data[:,0].argsort()]
    return data

_POOL = Pool(processes=4)

def patoms(frame: np.ndarray, pool=None) -> list[np.ndarray]:
    # reuse a single pool
    if pool is None:
        with Pool(processes=4) as p:
            data = p.apply(snapshot, (frame,))
    else:
        data = pool.apply(snapshot, (frame,))

    # split by colour gaps (vectorized)
    diffs = np.diff(data[:,0])
    splits = np.where(diffs > 0.008)[0]+1
    chunks = np.split(data, splits)
    # build normalized patoms (no Python‐loops over pixels)
    out = []
    for seg in chunks:
        x, y, c = seg[:,1], seg[:,2], seg[:,0]
        # fast vector stats
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        norm_x = 2*(x - x_min)/max(x_max-x_min,1) - 1
        norm_y = 2*(y - y_min)/max(y_max-y_min,1) - 1
        centroid = np.array([x.mean(), y.mean()])

        # build the 3 real columns
        pixel_data = np.column_stack((norm_x, norm_y, c))
        # pad a 4th column of nan
        pad = np.full((pixel_data.shape[0], 1), np.nan, dtype=pixel_data.dtype)
        pixel_data4 = np.hstack((pixel_data, pad))

        # single stack and dtype conversion
        patom = np.vstack([
            np.array([np.nan, *centroid, np.nan]),
            np.array([x_min, x_max, y_min, y_max]),
            pixel_data4
        ]).astype('float32')
        out.append(patom)

    return out