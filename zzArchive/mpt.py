# multiprocessing learning
import multiprocessing
# print(multiprocessing.cpu_count())

import numpy as np
from multiprocessing import Pool, Lock, Process, Queue, current_process
import time
import math
from operator import itemgetter
from time import perf_counter
import queue # imported for using queue.Empty exception
import sys
sys.path.append('/home/gerard/Desktop/capstone_project')

# import locally generated functions
from snapshot_3d_pattern_v5 import snapshot_pattern

np.random.seed(5555)
rand_array = np.random.random((30, 720, 1280))

def worker(array, start, end):
    """A worker function to calculate nearest neighbours."""
    for i in range(start, end):
        output = snapshot_pattern(array, i)
        return output

def main(array):
    threshold = 0.00005
    outs = []
    outs.append(worker)

    segment = 26 // 8
    processes = []

    for i in range(8):
        start = i * segment
        if i == 8 - 1:
            end = 25  # Ensure the last segment goes up to the end
        else:
            end = start + segment
        # Creating a process for each segment
        p = multiprocessing.Process(target=worker, args=(array, start, end))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    # combine the outputs of each nearest neighbour function
    combined_output = sorted(set([i for x in processes for i in x]))
    
    # split list when value between subsequent elements is greater than threshold
    res, last = [[]], None
    for x in combined_output:
        if last is None or abs(last - x[0]) <= threshold: #runtime warning here
            res[-1].append(x)
        else:
            res.append([x])
        last = x[0]

    # sort the lists of tuples based on the indices (need to get indices as tuple)
    s_res = []
    for i in res:
        s = sorted(i, key=itemgetter(1))
        if len(s) >= 10:
            s_res.append(s)

    # then need to obtain a normalised distance for all points from the 'center' of the pattern
    norm_patoms = []
    for pat in s_res:
        x = [p[1][0] for p in pat]
        min_x = min(x); max_x = max(x)
        norm_x = [(i - min_x)/(max_x - min_x) for i in x]
        y = [p[1][1] for p in pat]
        min_y = min(y); max_y = max(y)
        norm_y = [(i - min_y)/(max_y - min_y) for i in y]
        z = [p[1][2] for p in pat]
        min_z = min(z); max_z = max(z)
        norm_z = [(i - min_z)/(max_z - min_z) for i in z]
        centroid = list((sum(x) / len(pat), sum(y) / len(pat), sum(z) / len(pat)))
        centroid_norm = list((sum(norm_x) / len(pat), sum(norm_y) / len(pat), sum(norm_z) / len(pat)))
        centroid_list = [tuple([0.0]+centroid+centroid_norm+[0.0])]
        loc = [p[1] for p in pat]
        dist = list(map(lambda x: math.dist(centroid, list(x)), loc))
        tot = sum(dist)
        # normalised distance value is from centroid
        norm_dist = [i/tot for i in dist]
        val = [p[0] for p in pat]
        patom = list(zip(val, x, y, z, norm_x, norm_y, norm_z, norm_dist))
        patom = patom + centroid_list
        norm_patoms.append(patom)

    end_stack = perf_counter()

    stacked_time = end_stack - strt_stack
    print('Took this many seconds: ', stacked_time)

    return norm_patoms

if __name__ == '__main__':
    strt_stack = perf_counter()
    main(rand_array)

    end_stack = perf_counter()

    stacked_time = end_stack - strt_stack
    print('Took this many seconds: ', stacked_time)