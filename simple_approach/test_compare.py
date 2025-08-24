
import numpy as np
from multiprocessing import Pool, cpu_count
from time import perf_counter
import glob, os
from typing import List, Tuple, Union
import sys
from itertools import combinations
import matplotlib.pyplot as plt

sys.path.append('/home/gerard/Desktop/capstone_project')
sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach')

historic_data = sys.path.append('/home/gerard/Desktop/capstone_project/simple_approach/historic_data')
root = os.path.dirname(__file__)
folder = os.path.join(root, 'test_historic_data')
output = os.path.join(root, 'test_reference_patoms')
os.makedirs(output, exist_ok=True)

from tabula_rasa_technology.simple_approach.reference import create_reference_patom
from tabula_rasa_technology.simple_approach.patoms import patoms
from tabula_rasa_technology.simple_approach.unnormalise import unnormalise_xy

### number of sequences to import
# n = 5
# # Using the same input dataset as per the compartor CNN-LSTM model
# data = np.load('mnist_test_seq.npy')
# # Swap the axes representing the number of frames and number of data samples.
# dataset = np.swapaxes(data, 0, 1)
# # We'll pick out 1000 of the 10000 total examples and use those.
# dataset = dataset[:n, ...]
# print('loaded')
# patoms_to_save = []
# #generate patoms from sequences and save to disk
# for i in range(0,n,1):
#     print('sequence num:',i)
#     sequence = dataset[i]
#     for j in range(0,20,1):
#         frame = sequence[j]
#         out_patoms = patoms(frame)
#         for i in out_patoms:
#             # save patoms to disk
#             np.save(f'test_historic_data/test_patom_{str(i[0,0])}', i)
#         del out_patoms

# bin value of 64 as original image shape was 64x64. normalised patom values are less than this
# but bins with zero values are ignored
def radial_power_spectrum(array):
    # compute the 2d fourier transform and and recenter to make symmetric
    f = np.fft.fftshift(np.fft.fft2(array.astype(np.float32)))
    # calculate the power spectrum
    mag = np.abs(f)**2

    # set bins equal to the shape of the original array (arbitray)
    nbins=64
    # build radius map, centred at spectrum centre
    h, w = array.shape
    cy, cx = (h-1)/2.0, (w-1)/2.0
    # create open grids to match x, y coordinate number
    y, x = np.ogrid[:h, :w]
    # compute radial distance for each coordinate from the spectrum centre
    r = np.sqrt((y - cy)**2 + (x - cx)**2)

    # bin radial distances (normalised to ensure negative values are set between 0 and 1)
    r_norm = r / r.max()
    # created bins 
    bins = np.linspace(0, 1.0, nbins+1)
    # flatten the normalised radial distance array and assign each pixel to a bin
    idx = np.digitize(r_norm.ravel(), bins) - 1
    # sum the values per bin
    sums = np.bincount(idx, weights=mag.ravel(), minlength=nbins)
    # count how many pixels fall into each bin
    counts = np.bincount(idx, minlength=nbins)
    # compute radial average
    radial = np.divide(sums, np.maximum(counts, 1), dtype=np.float32)

    # https://stackoverflow.com/questions/49538185/purpose-of-numpy-log1p
    radial = np.log1p(radial)
    # l1 normalise so all bins sum to 1
    radial /= radial.sum() + 1e-8

    return radial

def rps_distance(array1, array2):
    a = radial_power_spectrum(array1)
    b = radial_power_spectrum(array2)
    
    # calculate the chi-squared distance between corresponding bins from each patom
    return 0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-8))

# colour distance 
def colour_histogram(col):
    
    # flatten array
    col = np.asarray(col).reshape(-1)

    # create number of bins (arbitrary)
    bins=32
    
    # put colours into bins
    h, _ = np.histogram(col, bins=bins, range=(1, 255))
    h = h.astype(np.float32)
    
    # l1 normalise so all bins sum to 1
    h /= (h.sum() + 1e-8)

    return h

# colour distance 
def col_distance(arr_col1, arr_col2):
    p = colour_histogram(arr_col1)
    q = colour_histogram(arr_col2)
    
    # calculate the chi-squared distance between corresponding bins from each patom
    return 0.5 * np.sum((p - q)**2 / (p + q + 1e-12), dtype=np.float32)

def compare(array1, array2):
    
    id1 = array1[0,0]; id2 = array2[0,0]
    coords1 = array1[2:,:2]
    coords2 = array2[2:,:2]
    cols1 = array1[2:,2]
    cols2 = array2[2:,2]
    coordinate_dist = rps_distance(coords1, coords2)
    colour_dist = col_distance(cols1, cols2)

    score = coordinate_dist + colour_dist

    return [id1, id2, score, coordinate_dist, colour_dist]


shape = (64, 64)

test_file_paths = glob.glob(os.path.join(folder, '*.npy'))
test_patoms = [np.load(i) for i in test_file_paths]

sim_threshold = 0.01
scores = []
for pat1, pat2 in combinations(test_patoms,2):
    if (pat1[2:,:].shape[0] > 0) and (pat2[2:,:].shape[0] > 0):
        id1, id2, score, coordinate_dist, colour_dist = compare(pat1, pat2)
        if (score < 0.05) and (pat1.shape[0] > 8): 
            #scores.append(score)
            print('pat1\n', pat1[2:,:3], '\n'); print('pat2\n', pat2[2:,:3])
            img_pat1 = pat1
            img_pat2 = pat2
            
            h, w = shape

            # reconstruct both patoms
            unnorm1 = unnormalise_xy(img_pat1)
            y1 = unnorm1[2:, 1].astype(np.int64, copy=False)
            x1 = unnorm1[2:, 0].astype(np.int64, copy=False)
            v1 = unnorm1[2:, 2].astype(np.int64, copy=False)
            out1 = np.full(shape, 0, dtype=int)
            out1[y1, x1] = v1 

            unnorm2 = unnormalise_xy(img_pat2)
            y2 = unnorm2[2:, 1].astype(np.int64, copy=False)
            x2 = unnorm2[2:, 0].astype(np.int64, copy=False)
            v2 = unnorm2[2:, 2].astype(np.int64, copy=False)
            out2 = np.full(shape, 0, dtype=int)
            out2[y2, x2] = v2

            cmap = 'gray' if img_pat1.ndim == 2 else None

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

            im = None
            for ax, img, title in zip(axes, (out1, out2), (f"Pat1 id:{id1:.8f}", f"Pat2 id:{id2:.8f}")):
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis('off')

            fig.suptitle(f'Sim Measures - Score:{score:.6f}, Coord:{coordinate_dist:.6f}, Col:{colour_dist:.6f}', fontsize=15)
    
            plt.show()

# max_score = max(scores)
# min_score = min(scores)
# avg_score = sum(scores)/len(scores)

# print('max:', max_score)
# print('min:', min_score)
# print('avg:', avg_score)