###### Part 1. ######
#### read in data ####
######################

## add ignore warnings for now, will remove and debug once full algorithm is complete
# import warnings
# warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter
import numpy as np
import sys

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

from tabula_rasa_technology.simple_approach.patoms import patoms, _POOL
from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.ref_patoms import create_reference_patom
from tabula_rasa_technology.simple_approach.ref_patom_mgmt import ArrayGroup, ArrayGroupManager

# -- instantiate the manager --
mgr = ArrayGroupManager(
    compare_fn=compare,
    inc_ref_fn=create_reference_patom,
    threshold=0.25
)

# number of sequences to import
n = 5
# Using the same input dataset as per the compartor CNN-LSTM model
data = np.load('mnist_test_seq.npy')
# Swap the axes representing the number of frames and number of data samples.
dataset = np.swapaxes(data, 0, 1)
# We'll pick out 1000 of the 10000 total examples and use those.
dataset = dataset[:n, ...]
print('loaded')
start = perf_counter()
def main():
    #generate patoms from sequences
    for i in range(0,n,1):
        print('seq num:', i, 'time start:', round((perf_counter()-start)/60,2))
        sequence = dataset[i]
        for j in range(0,20,1):
            frame = sequence[j] / 255.00
            
            ####### Part 2. #######
            #### create patoms ####
            #######################
            out_patoms = patoms(frame)

            ########## Part 3. ########
            #### create ref patoms ####
            ###########################
            for new_arr in out_patoms:
                mgr.add_array(new_arr)
        
        print('seq num:', i, 'time end:', round((perf_counter()-start)/60,2))  


    # 3) When you're 100% done with ALL patoms() calls:
    from patoms import _POOL
    _POOL.close()
    _POOL.join()

# -- get reference patom and write to disk --
all_groups = mgr.get_all_groups()
for idx, grp in enumerate(all_groups):
    ref = grp[0]
    np.save(f'reference_patoms/patom_{str(ref[0,0])}', ref)       

## create dictionaries for each sequence for reference patoms

if __name__ == "__main__":
    main()