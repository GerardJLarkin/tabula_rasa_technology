## add ignore warnings for now, will remove and debug once full algorithm is complete
# import warnings
# warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter
import numpy as np
import sys

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.patoms_v0 import patoms

# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:100, ...]
print('loaded')
#generate patoms from sequences and save to disk
seq_ind = 0
for i in range(0,100,1):
    print('sequnce num:',i)
    sequence = dataset[i]
    for j in range(0,20,1):
        frame = sequence[j]
        out_patoms = patoms(frame, seq_ind)
        for i in out_patoms:
            patom_id = i[0,0]
            # save patoms to disk
            np.save(f'historic_data/patom_{patom_id}', i)
        del out_patoms


end = perf_counter()
print("Time taken (mins):", (end - start)/60)