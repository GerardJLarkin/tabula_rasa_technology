import numpy as np
import os, glob
from time import perf_counter
import sys
from itertools import combinations

# ## append filepath to allow files to be called from within project folder
# sys.path.append('/home/gerard/Desktop/capstone_project')
# sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')
# historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
# root = os.path.dirname(os.path.abspath(__file__))
# folder = os.path.join(root, 'historic_data')

# from tabula_rasa_technology.simple_approach.reference import create_reference_patom

# # --- load reference patoms once ---
# reference_dir = os.path.join(root, 'reference_patoms')
# ref_patoms = [np.load(os.path.join(folder, fn),) for fn in os.listdir(folder)][80:100]

# ref_patom = create_reference_patom(ref_patoms)
# # Print grouped arrays
# # for idx, ref in enumerate(ref_patom):
# print('ref 1st row:', ref_patom[0,:])

import numpy as np

mask = np.array([[1, 3, 0, 0],
                 [0, 0, 0, 0],
                 [0, 10, 5, 0],
                 [0, 0, 0, 0]], dtype=np.float32)

f = np.fft.fftshift(np.fft.fft2(mask))
mag = np.abs(f)**2
print("power spectrum:", mag)

h, w = mask.shape
cy, cx = (h-1)/2.0, (w-1)/2.0  # center = (1.5, 1.5)
y, x = np.ogrid[:h, :w]
r = np.sqrt((y - cy)**2 + (x - cx)**2)

r_norm = r / r.max()
bins = np.linspace(0, 1.0, 10)  # e.g. 4 bins
idx = np.digitize(r_norm.ravel(), bins) - 1
print(idx)

print("radius map:\n", r, r_norm)
print("bin indices for each pixel:\n", idx.reshape(h,w))

sums = np.bincount(idx, weights=mag.ravel(), minlength=4)
print("bin sums:", sums)

#count how many pixels fall into each bin
counts = np.bincount(idx, minlength=10)
print('counts', counts)
# compute radial average
radial = np.divide(sums, np.maximum(counts, 1), dtype=np.float32)
print(radial)
      
radial = np.log1p(radial)
print(radial)
# l1 normalise so all bins sum to 1
radial /= radial.sum() + 1e-8
print(radial)

