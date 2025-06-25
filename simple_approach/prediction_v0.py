## prediction (maybe)
import numpy as np
import os
import glob
import sys
from time import perf_counter
from typing import Iterable, Callable, Generator, Any, Tuple
from typing import Callable, List, Tuple, Set
from itertools import product, islice
import pickle

start = perf_counter()

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')

from tabula_rasa_technology.simple_approach.compare_v0 import ref_compare, compare
from tabula_rasa_technology.simple_approach.patoms_v0 import patoms

root = os.path.dirname(os.path.abspath(__file__))

with open(root+'/vrlp.pkl', 'rb') as fp:
    vrlp = pickle.load(fp)

print(dict(islice(vrlp.items(), 3)))

with open(root+'/vrlv.pkl', 'rb') as fv:
    vrlv = pickle.load(fv)

print(dict(islice(vrlv.items(), 3)))

cnt_ = 0
while cnt_ <= 100:
    for key, value in vrlv.items():
        if value > 0:
            print(key, value)
            cnt_ += 1