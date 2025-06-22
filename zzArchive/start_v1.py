################################################################################
# write commentary and pseudo code on how the algorithm should be structured/  #
# work, etc.                                                                   #
################################################################################
from time import perf_counter, sleep, gmtime, strftime, localtime
import numpy as np
import random
from operator import itemgetter
import math
import datetime
import csv
#from memory_profiler import profile
from tempfile import TemporaryFile
norm_file = TemporaryFile()

np.random.seed(5555)

import sys
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/initialdb')

# import locally generated functions
from snapshot_3d_pattern_v4 import snapshot_pattern
#from create_database import InitialDB as idb

operating_time = strftime("%H:%M:%S", localtime())
timestamp = datetime.datetime.now()

strt = perf_counter()
time_array = np.load('/home/gerard/Desktop/capstone_project/norm_file.npy')
#print((time_array[:2,:2,:2]))

patoms = snapshot_pattern(time_array)

# # save list of list to file in what way?
# # save each patom to a separate file
# # structure each file with the pixel value, inde, etc. as a single row
# for ind, pat in enumerate(patoms):
#     with open(f'/home/gerard/Desktop/capstone_project/patoms/pat{ind}.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',')
#         writer.writerows(pat)
end = perf_counter()

print('Run time: ',end-strt)



















### simulate continuous data feed (streaming data)
# while True:
#     if operating_time >= '07:00:00':
#         print('time to observe')#### CPU 2 will accept a new snapshot every 1/30th of a second (30 snapshots per second) ####
#         stack = []
#         loop_op_time = strftime("%H:%M:%S", localtime())
#         #### CPU 1 will feed in the following acquired inputs to CPU2 ####
#         # take in data from cameras
#         cam1a = np.random.random((1280, 720))
#         cam1b = np.random.random((1280, 720))
#         cam1c = np.random.random((1280, 720))
#         cam1 = np.add(cam1a, cam1b, cam1c)
#         cam2a = np.random.random((1280, 720))
#         cam2b = np.random.random((1280, 720))
#         cam2c = np.random.random((1280, 720))
#         cam2 = np.add(cam2a, cam2b, cam2c)
        
#         # # take in data from microphones
#         # mic1 = np.random.random(20000).tolist()
#         # mic2 = np.random.random(20000).tolist()
#         # mic3 = np.random.random(20000).tolist()
#         # mic4 = np.random.random(20000).tolist()
#         # mic5 = np.random.random(20000).tolist()
#         # mic6 = np.random.random(20000).tolist()

#         # # take in data from gyroscpe
#         # gyr1 = np.random.random(3).tolist()

#         # # take in data from acceleromoter
#         # acc1 = np.random.random(3).tolist()

#         # data will be fed into this loop 30 snapshots each second
#         # combine all data inputs (data inputs have been vectorised)
#         snapshot = np.concatenate((cam1, cam2)) # + mic1 + mic2 + mic3 + mic4 + mic5 + mic6 + gyr1 + acc1 
#         print(snapshot.shape)
#         #### CPU 2 will accept a new snapshot every 1/30th of a second (30 snapshots per second) ####

#         # identify pattern in snapshot
#         # types of 2d patterns dots, lines, waves, arcs (of dots, lines, waves), planes, arc planes (of dots lines, waves)
#         # compare a single pixel to its 8 nearest neighbours, if neighbour is within a threshold difference they start a pattern
#         # then commbine all patterns that are adjacent and are within the threshold difference value
#         patoms = snapshot_pattern(snapshot)
        
    
#         # start a loop to take in each snapshot as it is generated, once the time 
#         # for a stacked window has elapsed and all the data has been added to a single
#         # matrix, reset the timer and start adding the next set of snapshot instances
#         # to a new matrix
#         stacked_time = 0
#         while stacked_time < 1.0:
#             strt_snap = perf_counter()
#             stack.append(patoms)
#             sleep(1/30)
#             end_snap = perf_counter()
#             stacked_time += end_snap - strt_snap
#             print(stacked_time)
#             if stacked_time >= 30.0:
#                 stacked_time == 0
#         #stack_array = np.array(stack)
#         print(stack[0][5:6])
#         del stack

#     # break loop to go into dreamng period or just redirect loop to dreaming algorithm
#     elif operating_time >= '20:00:00':
#         print('time to dream')
#         break

#     else:
#         pass
        
