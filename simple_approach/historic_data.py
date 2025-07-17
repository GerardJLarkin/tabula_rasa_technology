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

from tabula_rasa_technology.simple_approach.patoms_v1 import patoms

# number of sequences to import
n = 100
# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:n, ...]
print('loaded')
#generate patoms from sequences and save to disk
for i in range(0,n,1):
    print('sequence num:',i)
    sequence = dataset[i]
    for j in range(0,20,1):
        frame = sequence[j] / 255.00
        out_patoms = patoms(frame)
        for i in out_patoms:
            # save patoms to disk
            np.save(f'historic_data/patom_{str(i[0,0])}', i)
        del out_patoms

end = perf_counter()
print("Time taken (mins):", (end - start)/60)