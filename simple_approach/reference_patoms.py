## create reference patoms
import numpy as np
import os, glob
from time import perf_counter
import sys

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(os.path.abspath(__file__))

from tabula_rasa_technology.simple_approach.compare import compare
from tabula_rasa_technology.simple_approach.patom_groups import group_arrays_multiprocess, load_memmap

def main():
    start = perf_counter()
    root = os.path.dirname(__file__)
    folder = os.path.join(root, 'historic_data')
    output = os.path.join(root, 'reference_patoms')
    os.makedirs(output, exist_ok=True)

    file_paths = glob.glob(os.path.join(folder, '*.npy'))
    sim_threshold = 0.20

    reference_patoms = group_arrays_multiprocess(file_paths, compare, sim_threshold)
    
    for ref in reference_patoms:
        np.save(os.path.join(output, f'patom_{ref[0,0]:.8f}.npy'), ref)

    total = (perf_counter() - start) / 60
    print(f"Total elapsed: {total:.2f} minutes")

if __name__ == '__main__':
    main()
