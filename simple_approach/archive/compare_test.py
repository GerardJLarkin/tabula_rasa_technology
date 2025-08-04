import numpy as np
from typing import Any, List, Union

from numba import njit

## Part 2: function to compare numpy arrays
@njit(fastmath=True)
def compare_jit(a_vals, a_xy, b_vals, b_xy) -> float:
    # a_vals, b_vals: colour arrays (m,) and (n,)
    # a_xy,   b_xy:   position arrays (m,2) and (n,2)
    m, n = a_vals.shape[0], b_vals.shape[0]
    # fill similarity
    fill = abs(m-n)/((m+n)/2)
    # compute pairwise pos distances
    min_d=1e9; max_d=0.0; sum_d=0.0
    for i in range(m):
        for j in range(n):
            dx = a_xy[i,0] - b_xy[j,0]
            dy = a_xy[i,1] - b_xy[j,1]
            d = (dx*dx + dy*dy)**0.5
            if d<min_d: min_d=d
            if d>max_d: max_d=d
            sum_d += d
    # normalized pos sim
    denom = max(max_d-min_d, 1.0)
    pos_sim = (sum_d/m/n - min_d)/denom
    # colour sim
    col_diff = 0.0
    for i in range(m):
        for j in range(n):
            col_diff += abs(a_vals[i] - b_vals[j])
    col_sim = col_diff/(m*n)
    
    # weighted sum
    return ((pos_sim*4) + (col_sim*1) + (min(fill,1.0)*2)) / 7.0



def compare(a: np.ndarray, b: np.ndarray):
    # unpack outside JIT
    id_a, id_b = a[0,0], b[0,0]
    a_vals = a[2:,2].astype(np.float32, copy=False)
    b_vals = b[2:,2].astype(np.float32, copy=False)
    a_xy   = a[2:,:2].astype(np.float32, copy=False)
    b_xy   = b[2:,:2].astype(np.float32, copy=False)
    sim = compare_jit(a_vals, a_xy, b_vals, b_xy)
    
    return [id_a, id_b, sim]